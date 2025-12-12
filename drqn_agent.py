import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DRQN(nn.Module):
    """
    Deep Recurrent Q-Network (DRQN) Architecture.
    
    Structure:
    1. FC Layer: Embeds the input observation (choice + reward).
    2. LSTM Layer: Maintains the hidden belief state over time.
    3. Output Layer: Maps LSTM output to Q-values for each action.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DRQN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. Input Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # 2. Recurrent Layer (LSTM)
        # batch_first=True expects input shape (batch, seq_len, features)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Q-Value Output Head
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Tuple (h_0, c_0) for LSTM initialization.
            
        Returns:
            q_values: (batch_size, seq_len, output_dim)
            next_hidden: (h_n, c_n)
        """
        # Activation for the encoder
        x = F.relu(self.fc1(x))
        
        # Pass through LSTM
        # If hidden is None, it defaults to zeros
        lstm_out, next_hidden = self.lstm(x, hidden)
        
        # Calculate Q-values from LSTM output
        q_values = self.fc_out(lstm_out)
        
        return q_values, next_hidden

    def init_hidden(self, batch_size, device):
        """Helper to initialize zero hidden states."""
        return (torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device))


class RecurrentReplayBuffer:
    """
    Replay Buffer for DRQN.
    Stores entire episodes (or chunks of sequences) to train the LSTM.
    """
    def __init__(self, capacity, sequence_length):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def push(self, state, action, reward, next_state, done):
        """
        Stores a single step. 
        Note: For DRQN, we often reconstruct sequences during sampling or 
        store them as lists of steps. Here we store individual steps and 
        will reconstruct sequences in the `sample` method.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of valid sequences.
        
        Strategy:
        - Randomly pick 'batch_size' starting indices from the buffer.
        - For each index, grab 'sequence_length' consecutive steps.
        - Pad with zeros if a sequence hits the buffer boundaries or an episode end.
        """
        # We need a robust way to sample sequences. 
        # For simplicity in this example, we sample random start points.
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
        
        # Valid starting indices (must have room for seq_len)
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        
        for idx in indices:
            seq_state, seq_action, seq_reward, seq_next_state, seq_done = [], [], [], [], []
            
            for i in range(self.sequence_length):
                # Check if index exists
                if idx + i < len(self.buffer):
                    s, a, r, ns, d = self.buffer[idx + i]
                    
                    seq_state.append(s)
                    seq_action.append(a)
                    seq_reward.append(r)
                    seq_next_state.append(ns)
                    seq_done.append(d)
                    
                    # If this step finished the episode, stop adding to this sequence 
                    # (or we act as if the rest is padded/masked)
                    if d:
                        break
                else:
                    break
            
            # Pad sequences that are shorter than sequence_length
            actual_len = len(seq_state)
            pad_len = self.sequence_length - actual_len
            
            if pad_len > 0:
                # Padding with zeros (assuming state dim is 3)
                zero_state = np.zeros_like(seq_state[0])
                
                seq_state += [zero_state] * pad_len
                seq_action += [0] * pad_len
                seq_reward += [0.0] * pad_len
                seq_next_state += [zero_state] * pad_len
                seq_done += [True] * pad_len # Treat padding as 'done'
                
            batch_state.append(seq_state)
            batch_action.append(seq_action)
            batch_reward.append(seq_reward)
            batch_next_state.append(seq_next_state)
            batch_done.append(seq_done)
            
        return (np.array(batch_state), np.array(batch_action), 
                np.array(batch_reward), np.array(batch_next_state), 
                np.array(batch_done))

    def __len__(self):
        return len(self.buffer)
