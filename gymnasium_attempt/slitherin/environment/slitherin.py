from typing import Optional, TypedDict
import numpy as np
import gymnasium as gym
import random

from .snake import Snake


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
        self.apple = np.array([0, 0])  # Initialize with default, will be properly set by _spawn_apple
        self._spawn_apple()

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
            shape=(num_agents, grid_size[0], grid_size[1]),
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
        observations = self._get_obs()
        info = self._get_info()

        return observations, info

    
    def _get_occupied_postions(self):
        _occupied_positions = set()

        # Snake bodies
        for _snake in self.snakes:
            for _segment in _snake.body:
                _occupied_positions.add(tuple(_segment))

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

    def _get_obs(self) -> list[np.ndarray]:
        observations = []

        for view_idx in range(self.num_agents):
            obs = np.zeros(self.grid_size, dtype=np.float32)

            # Skip observation generation for dead snakes
            if self.snakes[view_idx].dead:
                observations.append(obs)  # Return empty observation for dead snakes
                continue

            for i, snake in enumerate(self.snakes):
                if snake.dead:
                    continue  # Skip dead snakes
                    
                for j, segment in enumerate(snake.body):
                    y, x = segment
                    if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                        if i == view_idx:
                            obs[y, x] = 2 if j == 0 else 1  # Own head/body
                        else:
                            obs[y, x] = -2 if j == 0 else -1  # Other snakes

            # Apple is same for everyone
            if self.apple is not None:
                apple_y, apple_x = self.apple
                if 0 <= apple_y < self.grid_size[0] and 0 <= apple_x < self.grid_size[1]:
                    obs[apple_y, apple_x] = 3

            observations.append(obs)

        return np.array(observations, dtype=np.float32)
    
    def _get_info(self) -> dict:
        # Count living snakes
        living_snakes = sum(1 for snake in self.snakes if not snake.dead)
        snake_lengths = [len(snake.body) for snake in self.snakes]
        
        return {
            "num_agents": self.num_agents,
            "living_snakes": living_snakes,
            "snake_lengths": snake_lengths,
            "dead_snakes": [i for i, snake in enumerate(self.snakes) if snake.dead]
        }
    
    def step(self, actions: list[int]):
        rewards = [self.rewards["idle"]] * self.num_agents
        terminated = [False] * self.num_agents
        truncated = [False] * self.num_agents

        new_heads = []
        head_positions = {}
        
        # 1. Move all living snakes
        for idx, (snake, action) in enumerate(zip(self.snakes, actions)):
            if snake.dead:
                terminated[idx] = True
                continue
                
            if abs(action - snake.direction) == 2:
                action = snake.direction  # Prevent reversing
            snake.move(action)
            head = tuple(snake.body[0])
            new_heads.append(head)
            head_positions.setdefault(head, []).append(idx)

        # 2. Grow if on apple
        for idx, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            if self.apple is not None and np.array_equal(snake.body[0], self.apple):
                rewards[idx] = self.rewards["win"]
                tail = snake.body[-1]
                snake.body = np.vstack([snake.body, tail[None, :]])  # Grow
                self._spawn_apple()

        # 3. Get updated occupied positions (includes bodies and walls)
        occupied_positions = self._get_occupied_postions()

        # 4. Collision Detection
        for idx, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            head = tuple(snake.body[0])

            # Head-on collision
            if head in head_positions and len(head_positions[head]) > 1:
                terminated[idx] = True
                rewards[idx] = self.rewards["lose"]
                snake.dead = True

            # Collision with wall or another snake's body
            elif head in occupied_positions:
                # Check if it's not colliding with its own body (first segment is head)
                if head not in [tuple(segment) for segment in snake.body[1:]]:
                    terminated[idx] = True
                    rewards[idx] = self.rewards["lose"]
                    snake.dead = True
        
        observations = self._get_obs()
        info = self._get_info()

        return observations, rewards, terminated, truncated, info