from typing import Optional, TypedDict
import numpy as np
import gymnasium as gym
import random

from snake import Snake


class RewardDict(TypedDict):
    win: int
    idle: int
    lose: int

class Slitherin(gym.Env):

    def __init__(
            self, 
            render_mode: Optional[str] = None, 
            grid_size: Optional[tuple] = (5, 5), # min is (4, 4)
            rewards: Optional[RewardDict] = {"win": 1, "idle": 0, "lose": -1},
            num_agents: Optional[int] = 2
        ):
        
        # The size of the square grid
        self.grid_size = grid_size

        # Number of agents
        self.num_agents = num_agents

        # Define Snakes
        self.snakes = [
            Snake(
                start_pos=(random.randint(0, grid_size[0] - 2), random.randint(0, grid_size[1] - 2)),
                starting_direction=random.randint(0, 3),  # 0=right, 1=up, 2=left, 3=down
                color=random.randint(0, 5)
            )
            for _ in range(num_agents)
        ]

        # Define first apple
        self._spawn_apple()

        self.apple = np.array()

        # Define reward
        self.rewards = rewards

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        # The observation is a a box where
        # 0 = nothing
        # 1 = OWN snake body
        # -1 = OTHER snake body
        # 2 = OWN Snake head
        # -2 = OTHER snake head
        # 3 = Apple
        # -3 = Wall (does not appear but reserved)
        self.observation_space = gym.spaces.Box(
            low=-3,
            high=3,
            shape=grid_size,
            dtype=np.float32
        )


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the snakes and apple positions
        self.snakes = [
            Snake(
                start_pos=(random.randint(0, self.grid_size[0] - 2), random.randint(0, self.grid_size[1] - 2)),
                starting_direction=random.randint(0, 3),  # 0=right, 1=up, 2=left, 3=down
                color=random.randint(0, 5)
            )
            for _ in range(self.num_agents)
        ]
        self._spawn_apple()

        # Get observation
        observation = self._get_obs()

        return observation

    
    def _get_occupied_postions(self):
        _occupied_positions = set()

        # Snake bodies
        for _snake in self.snakes:
            for _segment in _snake.body:
                _occupied_positions.add(_segment)

        # Walls
        for x in range(-1, self.grid_size[0] + 1):
            _occupied_positions.add((x, -1))
            _occupied_positions.add((x, self.grid_size[1]))
        for y in range(-1, self.grid_size[1] + 1):
            _occupied_positions.add((-1, y))
            _occupied_positions.add((self.grid_size[0], y))

        return _occupied_positions
    
    def _spawn_apple(self):
        # Get all the available positions by removing the occupied ones
        _all_available_positions = [
            (x, y)
            for x in range(self.grid_size[0])
            for y in range(self.grid_size[1])
            if (x, y) not in self._get_occupied_postions()
        ]

        if _all_available_positions:
            # Choose one positions from all available ones
            self.apple = np.array(random.choice(_all_available_positions))
        else:  
            # If no space is left
            self.apple = None

    def _get_obs(self):
        obs = np.zeros(self.grid_size, dtype=np.float32)

        for i, snake in enumerate(self.snakes):
            for j, segment in enumerate(snake.body):
                y, x = segment
                if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                    if i == 0:
                        obs[y, x] = 2 if j == 0 else 1  # Own head/body
                    else:
                        obs[y, x] = -2 if j == 0 else -1  # Other snakes

        # Place the apple
        apple_y, apple_x = self.apple
        if 0 <= apple_y < self.grid_size[0] and 0 <= apple_x < self.grid_size[1]:
            obs[apple_y, apple_x] = 3

        return obs
    
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