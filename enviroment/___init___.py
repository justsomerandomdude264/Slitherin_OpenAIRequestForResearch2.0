import gymnasium as gym
from enviroment.env import Slitherin
gym.register(
    id="gymnasium_env/Slitherin-V0",
    entry_point=Slitherin,
)