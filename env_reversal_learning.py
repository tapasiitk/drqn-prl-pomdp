import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ProbabilisticReversalLearningEnv(gym.Env):
    """
    Probabilistic Reversal Learning (PRL) Environment.
    
    Task Description:
        - Two options: Left (0) and Right (1).
        - One option is 'correct' (high reward prob), the other 'incorrect' (low reward prob).
        - The mapping reverses after a fixed criterion of consecutive correct choices.
    
    Observation Space:
        - Vector of size 3: [prev_choice_left, prev_choice_right, prev_reward]
          - prev_choice_left: 1 if last choice was Left, else 0
          - prev_choice_right: 1 if last choice was Right, else 0
          - prev_reward: 1 if last choice was rewarded, else 0
        
    Action Space:
        - Discrete(2): 0 = Left, 1 = Right
    """
    def __init__(self):
        super(ProbabilisticReversalLearningEnv, self).__init__()
        
        # Actions: 0 = Left, 1 = Right
        self.action_space = spaces.Discrete(2)
        
        # Observation: [one-hot action (2), previous reward (1)]
        # We use a Box space for the input vector.
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Task Parameters (from README Section 2)
        self.prob_correct = 0.8       # Reward probability for better option
        self.prob_incorrect = 0.2     # Reward probability for worse option
        self.reversal_criterion = 5   # Consecutive correct choices to trigger reversal
        
        # State variables
        self.correct_option = None    # 0 or 1
        self.consecutive_correct_count = 0
        
        # Keep track of last step for observation
        self.last_action = None
        self.last_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly choose which option is initially correct (0 or 1)
        self.correct_option = self.np_random.integers(0, 2)
        self.consecutive_correct_count = 0
        
        # Reset history
        self.last_action = None
        self.last_reward = 0.0
        
        # Initial observation: [0, 0, 0] (No previous action/reward)
        obs = np.zeros(3, dtype=np.float32)
        
        return obs, {}

    def step(self, action):
        # 1. Determine correctness of the choice (latent state check)
        is_choice_better = (action == self.correct_option)
        
        # 2. Determine reward probability based on latent state
        if is_choice_better:
            p_reward = self.prob_correct
        else:
            p_reward = self.prob_incorrect
            
        # 3. Sample Reward
        reward = 1.0 if self.np_random.random() < p_reward else 0.0
        
        # 4. Update Reversal Logic
        if is_choice_better:
            self.consecutive_correct_count += 1
        else:
            # Reset counter if they chose the worse option
            self.consecutive_correct_count = 0
            
        # Check if criterion is met
        reversal_occurred = False
        if self.consecutive_correct_count >= self.reversal_criterion:
            # Reverse the contingencies
            self.correct_option = 1 - self.correct_option
            self.consecutive_correct_count = 0
            reversal_occurred = True
            
        # 5. Construct Next Observation
        # Encode action as one-hot: Left=[1,0], Right=[0,1]
        obs_action = np.zeros(2, dtype=np.float32)
        obs_action[action] = 1.0
        
        # Concatenate [one-hot action, reward]
        obs = np.concatenate([obs_action, [reward]]).astype(np.float32)
        
        # Update history
        self.last_action = action
        self.last_reward = reward
        
        # 6. Return step info
        # This is a continuing task, so we don't naturally terminate. 
        # Limits will be handled by the training loop (TimeLimit wrapper).
        terminated = False
        truncated = False
        
        info = {
            "correct_option": self.correct_option,
            "reversal_occurred": reversal_occurred,
            "is_choice_better": is_choice_better
        }
        
        return obs, reward, terminated, truncated, info
