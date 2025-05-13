import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, grid_size, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = grid_size[0] * grid_size[1] * 32
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
        
    def forward(self, x):
        x = x.view(-1, 1, x.size(1), x.size(2))  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Replay Memory
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, grid_size, num_actions, lr, mem_size, eps_start, eps_end, eps_decay, batch_size, gamma):
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.policy_net = DQN(grid_size, num_actions).to(device)
        self.target_net = DQN(grid_size, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(mem_size)
        
        self.eps_threshold = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.steps_done = 0
    
    def select_action(self, state, eval_mode=False):
        # Epsilon-greedy policy
        sample = random.random()
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if eval_mode or sample > self.eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
        else:
            return random.randint(0, self.num_actions - 1)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors for each element in the batch
        state_batch = torch.from_numpy(np.array(batch.state)).float().to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        
        # Filter out None values in next_state_batch (terminal states)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                      device=device, dtype=torch.bool)
        non_final_next_states_np = np.array([s for s in batch.next_state if s is not None])
        non_final_next_states = torch.from_numpy(non_final_next_states_np).float().to(device)
        
        # Compute current Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values using target network
        next_state_values = torch.zeros(self.batch_size, device=device)
        if any(non_final_mask):
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_model': self.policy_net.state_dict(),
            'target_model': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_model'])
        self.target_net.load_state_dict(checkpoint['target_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])