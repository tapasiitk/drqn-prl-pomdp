import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from env_reversal_learning import ProbabilisticReversalLearningEnv
from drqn_agent import DRQN, RecurrentReplayBuffer

# --- Hyperparameters ---
BATCH_SIZE = 32
LR = 1e-3
GAMMA = 0.99            # Discount factor
EPS_START = 1.0         # Initial exploration rate
EPS_END = 0.05          # Final exploration rate
EPS_DECAY = 10000       # Decay rate (in steps)
TARGET_UPDATE = 1000    # How often to update target network
BUFFER_SIZE = 5000      # Replay buffer capacity (number of steps)
SEQ_LENGTH = 10         # Sequence length for LSTM training
HIDDEN_DIM = 64         # LSTM hidden size
EPISODES = 2000         # Total training episodes

# Device configuration (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    """
    Main training loop for the DRQN agent.
    """
    
    # 1. Initialize Environment
    env = ProbabilisticReversalLearningEnv()
    input_dim = env.observation_space.shape[0]  # Size 3: [L, R, reward]
    output_dim = env.action_space.n             # Size 2: [Left, Right]
    
    # 2. Initialize Networks
    # Policy Net: The network we train
    policy_net = DRQN(input_dim, args.hidden_dim, output_dim).to(device)
    # Target Net: A stable copy for calculating TD targets
    target_net = DRQN(input_dim, args.hidden_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target net is only for inference, no gradients

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 3. Initialize Replay Buffer
    # We store sequences to train the recurrent layer
    memory = RecurrentReplayBuffer(capacity=BUFFER_SIZE, sequence_length=SEQ_LENGTH)
    
    # Tracking metrics
    steps_done = 0
    all_rewards = []
    losses = []

    print(f"Starting training on {device}...")

    # --- Episode Loop ---
    for i_episode in range(args.episodes):
        
        # Reset environment and hidden state for new episode
        obs, _ = env.reset()
        # Initialize LSTM hidden state to zeros (1, 1, hidden_dim)
        hidden = policy_net.init_hidden(batch_size=1, device=device)
        
        total_reward = 0
        done = False
        episode_steps = 0  

        # We need to keep a temporary buffer for the current episode 
        # to push to replay memory step-by-step
        
        while not done:
            # --- Action Selection (Epsilon-Greedy) ---
            epsilon = EPS_END + (EPS_START - EPS_END) * \
                      np.exp(-1. * steps_done / EPS_DECAY)
            
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1).to(device)
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                # Even if random, we must pass input through LSTM to update hidden state
                with torch.no_grad():
                    _, hidden = policy_net(obs_tensor, hidden)
            else:
                with torch.no_grad():
                    q_values, hidden = policy_net(obs_tensor, hidden)
                    action = q_values.argmax().item()
            
            # # --- Step Environment ---
            # next_obs, reward, terminated, truncated, _ = env.step(action)
            # done = terminated or truncated
            
            # # Artificial termination for training stability if episode gets too long
            # # (Optional, but good for PRL if agent gets stuck)
            # if total_reward < -50: 
            #     done = True

            # ... inside the "while not done:" loop ...
        
            # --- Step Environment ---
            next_obs, reward, terminated, truncated, _ = env.step(action)
        
            # # FIX: Add a step limit per episode
            # episode_steps = 0 # Initialize this before the while loop!
            
            # Inside loop, increment:
            episode_steps += 1
            
            # Force done if too many steps (e.g., 200 trials per block)
            if episode_steps >= 200:
                truncated = True
                
            done = terminated or truncated
            
            # Remove the "total_reward < -50" check, it is not needed if we have the step limit.
            
            # Store transition in memory
            # Note: We store individual steps. The buffer handles sequence reconstruction.
            memory.push(obs, action, reward, next_obs, done)
            
            # Move to next state
            obs = next_obs
            total_reward += float(reward)
            steps_done += 1

            # --- Optimize Model ---
            if len(memory) > BATCH_SIZE:
                optimize_model(policy_net, target_net, memory, optimizer, criterion)

            # --- Update Target Network ---
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(total_reward)
        
        # Logging
        if i_episode % 100 == 0:
            avg_rew = np.mean(all_rewards[-100:])
            print(f"Episode {i_episode} | Avg Reward (last 100): {avg_rew:.2f} | Epsilon: {epsilon:.3f}")

    # --- Save Final Model ---
    if not os.path.exists('models'):
        os.makedirs('models')
    save_path = os.path.join('models', 'drqn_final.pt')
    torch.save(policy_net.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    
    # Simple plot of results
    plt.plot(all_rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig('results_training.png')


def optimize_model(policy_net, target_net, memory, optimizer, criterion):
    """
    Performs a single gradient descent step.
    """
    # 1. Sample a batch of sequences
    # Shapes: (batch_size, seq_len, feature_dim)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(BATCH_SIZE)

    # Convert to tensors
    state_batch = torch.tensor(state_batch, dtype=torch.float32).to(device)
    action_batch = torch.tensor(action_batch, dtype=torch.int64).to(device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(device)
    done_batch = torch.tensor(done_batch, dtype=torch.float32).to(device)

    # 2. Compute Q(s_t, a) - The Q-values for the actions we actually took
    # Initialize hidden state for the batch (starts at zero for the sampled sequence)
    # Note: A more advanced implementation might store hidden states in replay, 
    # but zero-initialization is standard for DRQN with random replay.
    hidden = policy_net.init_hidden(BATCH_SIZE, device)
    
    # Forward pass: Get Q-values for all steps in the sequence
    q_values, _ = policy_net(state_batch, hidden) 
    
    # Gather the Q-value corresponding to the specific action taken at each step
    # action_batch shape: (batch, seq) -> unsqueeze to (batch, seq, 1)
    q_values = q_values.gather(2, action_batch.unsqueeze(2)).squeeze(2)

    # 3. Compute V(s_{t+1}) for target - The Max Q-value for the next state
    with torch.no_grad():
        hidden_target = target_net.init_hidden(BATCH_SIZE, device)
        target_q_values, _ = target_net(next_state_batch, hidden_target)
        # Max Q-value over actions (dim 2)
        next_state_values = target_q_values.max(2)[0]

    # 4. Compute Expected Q values (Bellman Equation)
    # expected_Q = reward + gamma * max(Q(next_state)) * (1 - done)
    expected_q_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    # 5. Compute Loss
    # We compute loss over the entire sequence
    loss = criterion(q_values, expected_q_values)

    # 6. Optimize
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent exploding gradients in LSTM
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRQN on PRL Task")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of LSTM")
    args = parser.parse_args()
    
    train(args)
