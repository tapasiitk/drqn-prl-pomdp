import argparse
import torch
import numpy as np
import pandas as pd
import pickle
import os

from env_reversal_learning import ProbabilisticReversalLearningEnv
from drqn_agent import DRQN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, input_dim, hidden_dim, output_dim):
    """Loads the trained DRQN model."""
    model = DRQN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model

def run_evaluation(agent, env, num_episodes=100):
    """
    Runs the agent in the environment to collect behavioral data.
    Returns a list of DataFrames (one per episode).
    """
    all_data = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        hidden = agent.init_hidden(batch_size=1, device=device)
        done = False
        
        # Storage for this episode
        episode_history = {
            "trial": [],
            "choice": [],
            "reward": [],
            "correct_option": [],
            "is_correct_choice": [],
            "reversal_occurred": []
        }
        
        trial_count = 0
        
        while not done:
            # Prepare input
            obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1).to(device)
            
            # Select Action (Greedy for evaluation)
            with torch.no_grad():
                q_values, hidden = agent(obs_tensor, hidden)
                action = q_values.argmax().item()
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record Data
            episode_history["trial"].append(trial_count)
            episode_history["choice"].append(action)
            episode_history["reward"].append(reward)
            episode_history["correct_option"].append(info["correct_option"])
            episode_history["is_correct_choice"].append(1 if info["is_choice_better"] else 0)
            episode_history["reversal_occurred"].append(1 if info["reversal_occurred"] else 0)
            
            obs = next_obs
            trial_count += 1
            
            # Safety break for very long eval episodes
            if trial_count > 200: 
                done = True

        # Convert to DataFrame
        df = pd.DataFrame(episode_history)
        df["episode"] = ep
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

def compute_metrics(df):
    """
    Computes behavioral metrics from the raw trial data.
    """
    print("Computing metrics...")
    
    # 1. Overall Accuracy
    accuracy = df["is_correct_choice"].mean()
    
    # 2. Win-Stay / Lose-Shift (WSLS)
    # We look at trial t and t+1
    # Win-Stay: Reward(t)=1 AND Choice(t+1)==Choice(t)
    # Lose-Shift: Reward(t)=0 AND Choice(t+1)!=Choice(t)
    
    # Shift columns to align t with t+1
    df["prev_choice"] = df.groupby("episode")["choice"].shift(1)
    df["prev_reward"] = df.groupby("episode")["reward"].shift(1)
    
    # Filter for valid previous trials
    valid_df = df.dropna(subset=["prev_choice", "prev_reward"])
    
    # Calculate probabilities
    wins = valid_df[valid_df["prev_reward"] == 1]
    losses = valid_df[valid_df["prev_reward"] == 0]
    
    win_stay_prob = (wins["choice"] == wins["prev_choice"]).mean()
    lose_shift_prob = (losses["choice"] != losses["prev_choice"]).mean()
    
    # 3. Switch Latency (Adaptation speed)
    # How many trials after a reversal does it take to switch to the new best option?
    # We identify trials where 'reversal_occurred' == 1, then count trials until 'is_correct_choice' is consistently high?
    # Simpler metric: Average accuracy in the 10 trials AFTER a reversal.
    
    # Get indices where reversal occurred
    rev_indices = df.index[df["reversal_occurred"] == 1].tolist()
    
    # Extract peri-reversal traces (e.g., -5 to +10 trials around reversal)
    traces = []
    window_before = 5
    window_after = 10
    
    for idx in rev_indices:
        # Check bounds
        if idx - window_before >= 0 and idx + window_after < len(df):
            # Check if this slice belongs to the same episode
            ep_id = df.loc[idx, "episode"]
            slice_df = df.loc[idx - window_before : idx + window_after]
            
            if (slice_df["episode"] == ep_id).all():
                # Correct choice (1) or incorrect (0)
                # Note: At idx (reversal), the correct option flips.
                # So if agent picked 'old correct', is_correct_choice is now 0.
                traces.append(slice_df["is_correct_choice"].values)

    if traces:
        avg_reversal_curve = np.mean(traces, axis=0)
    else:
        avg_reversal_curve = np.zeros(window_before + window_after + 1)

    metrics = {
        "accuracy": accuracy,
        "win_stay": win_stay_prob,
        "lose_shift": lose_shift_prob,
        "reversal_curve": avg_reversal_curve
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/drqn_final.pt", help="Path to saved model")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=100, help="Number of eval episodes")
    parser.add_argument("--output", type=str, default="results/agent_stats.pkl")
    args = parser.parse_args()

    # Setup
    env = ProbabilisticReversalLearningEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # Load
    agent = load_model(args.model_path, input_dim, args.hidden_dim, output_dim)
    
    # Run
    print(f"Running evaluation for {args.episodes} episodes...")
    data_df = run_evaluation(agent, env, args.episodes)
    
    # Analyze
    stats = compute_metrics(data_df)
    
    print(f"Results:")
    print(f"  Accuracy: {stats['accuracy']:.3f}")
    print(f"  Win-Stay: {stats['win_stay']:.3f}")
    print(f"  Lose-Shift: {stats['lose_shift']:.3f}")
    
    # Save
    if not os.path.exists('results'):
        os.makedirs('results')
        
    with open(args.output, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Stats saved to {args.output}")
