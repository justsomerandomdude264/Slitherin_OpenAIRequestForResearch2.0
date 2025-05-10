import gymnasium as gym
from slitherin.environment.slitherin import Slitherin  

# Register the Slitherin environment
gym.register(
    id="Slitherin-v0",
    entry_point="slitherin.environment.slitherin:Slitherin",
    max_episode_steps=1000,
)

gym.register(
    id="Slitherin-7x7-v0",
    entry_point="slitherin.environment.slitherin:Slitherin",
    max_episode_steps=1000,
    kwargs={
        "grid_size": (7, 7),
        "num_agents": 3
    }
)

gym.register(
    id="Slitherin-9x9-v0",
    entry_point="slitherin.environment.slitherin:Slitherin",
    max_episode_steps=1000,
    kwargs={
        "grid_size": (9, 9),
        "num_agents": 3
    }
)

print("Slitherin environments registered successfully!")
