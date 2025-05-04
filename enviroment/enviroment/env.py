from typing import Optional, TypedDict
import numpy as np
import gymnasium as gym


class RewardDict(TypedDict):
    win: int
    idle: int
    lose: int

class Slitherin(gym.Env):

    def __init__(self, render_mode: Optional[str] = None, 
                 grid_size: Optional[tuple] = (5, 5), 
                 rewards: Optional[RewardDict] = {"win": 1, "idle": 0, "lose": -1}):
        
        # The size of the square grid
        self.grid_size = grid_size

        # Define Snake and Apple position
        self.snake = np.array([[0, 0]])
        self.apple = np.array([0, 0])

        # Define reward
        self.rewards = rewards

        # The observation is a dict of the postions' of the snake and apple (x, y), (x, y)
        self.observation_space = gym.spaces.Dict(
            {
                "snake": gym.spaces.Box(low=0, high=max(self.grid_size), shape=(2,), dtype=np.int32),
                "apple": gym.spaces.Box(low=0, high=max(self.grid_size), shape=(2,), dtype=np.int32),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        # Define direction
        self.direction = self._action_to_direction[0] 

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the snake and apple positions
        self.snake = np.array([[0, 0]])
        self.apple = np.array([0, 0])

        # Reset direction and done
        self.direction = self._action_to_direction[0]
        self.done = False

        # Get observation
        observation = self._get_obs()

        return observation


    def _get_obs(self):
        head_y, head_x = self.snake[0]
        apple_y, apple_x = self.food
        return {
            "snake": np.array([head_y, head_x], dtype=np.int32),
            "apple": np.array([apple_y, apple_x], dtype=np.int32),
        }
    
    def _get_info(self):
         return {
            "distance": np.linalg.norm(
                self.snake[0] - self.apple, ord=1
            )
        }
    
    def step(self, action):

        if abs(action - self.direction) == 2:
            action = self.direction

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        head = self.snake[0]
        move = self._action_to_direction[action]
        new_head = self.head + move

        # Check for collisions
        if (not 0 <= new_head[0] < self.grid_size[0] or
            not 0 <= new_head[1] < self.grid_size[1] or
            new_head in self.snake):
            
            # If collided, then game is over
            observations = self._get_obs()
            reward = self.rewards["lose"]
            self.done = terminated = True
            info = self._get_info()
            truncated = False

            return observations, reward, terminated, truncated, info

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info