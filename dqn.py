import numpy as np
import random
import torch
from dqn_helper.func import train_dqn_self_play, evaluate_agents
import os

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("--- STARTING TRAINING ---")

# Define hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10
LEARNING_RATE = 0.0003
MEMORY_SIZE = 100000
NUM_EPISODES = 500
MODEL_PATH = 'models'
REWARDS = {"win": 1, "idle": 0, "lose": -1}
GRID_SIZE = (10, 10)
NUM_AGENTS = 3

EVAL_EPISODES = 10
EVAL_EPISODES = True

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)



if __name__ == "__main__":
    # Train
    train_dqn_self_play(NUM_EPISODES, MODEL_PATH, RANDOM_SEED, TARGET_UPDATE, LEARNING_RATE, MEMORY_SIZE, EPS_START, EPS_END, EPS_DECAY, BATCH_SIZE, GAMMA, REWARDS, GRID_SIZE, NUM_AGENTS)
    # Evaluate
    evaluate_agents(MODEL_PATH, RANDOM_SEED, REWARDS, NUM_AGENTS, GRID_SIZE, LEARNING_RATE, MEMORY_SIZE, EPS_START, EPS_END, EPS_DECAY, BATCH_SIZE, GAMMA, EVAL_EPISODES, EVAL_EPISODES)