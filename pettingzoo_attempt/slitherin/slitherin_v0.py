from typing import Optional, TypedDict, Dict
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
import random


class Snake:
    def __init__(self, start_pos, starting_direction, color):
        """Initialize snake with a starting position, direction, and color."""
        self.body = np.array([start_pos])  # Head is at index 0
        self.direction = starting_direction  # 0=right, 1=up, 2=left, 3=down
        self.color = color
        self.dead = False

    def move(self, action):
        """Move the snake according to the given action."""
        self.direction = action
        
        # Calculate the new head position based on the direction
        if self.direction == 0:    # right
            new_head = self.body[0] + [0, 1]
        elif self.direction == 1:  # up
            new_head = self.body[0] + [-1, 0]
        elif self.direction == 2:  # left
            new_head = self.body[0] + [0, -1]
        else:                      # down
            new_head = self.body[0] + [1, 0]
        
        # Move body segments (all except the head)
        self.body = np.vstack([new_head[None, :], self.body[:-1]])


class RewardDict(TypedDict):
    win: int
    idle: int
    lose: int


class Slitherin(ParallelEnv):
    metadata = {
        "name": "slitherin-v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
            self,
            grid_size: tuple = (5, 5),  # min is (4, 4)
            rewards: RewardDict = {"win": 1, "idle": 0, "lose": -1},
            num_agents: int = 2,
            render_mode: Optional[str] = None,
    ):
        # The size of the square grid
        self.grid_size = grid_size
        
        # Number of agents
        self._num_agents = num_agents
        
        # PettingZoo requires agent ids to be strings
        self.possible_agents = [f"snake_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()
        
        # Define reward
        self._rewards = rewards
        
        # Define observation and action spaces
        self.observation_spaces = {
            agent: Box(
                low=-3,
                high=3,
                shape=(grid_size[0], grid_size[1]),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }
        
        # Rendering setup
        self.render_mode = render_mode
        
        # Initialize the environment (will be properly set in reset)
        self.snakes = []
        self.apple = None
        self.agent_name_mapping = dict(zip(self.possible_agents, range(len(self.possible_agents))))

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset agents list to all possible agents
        self.agents = self.possible_agents.copy()
        
        # Reset the snakes
        self.snakes = [
            Snake(
                start_pos=(random.randint(0, self.grid_size[0] - 2), random.randint(0, self.grid_size[1] - 2)),
                starting_direction=random.randint(0, 3),  # 0=right, 1=up, 2=left, 3=down
                color=random.randint(0, 5)
            )
            for _ in range(self._num_agents)
        ]
        
        # Reset apple position
        self._spawn_apple()
        
        # Get observations and info
        observations = self._get_obs()
        infos = self._get_info()
        
        return observations, infos

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

    def _get_obs(self) -> Dict[str, np.ndarray]:
        observations = {}

        for agent_id in self.agents:
            agent_idx = self.agent_name_mapping[agent_id]
            obs = np.zeros(self.grid_size, dtype=np.float32)

            # Skip observation generation for dead snakes
            if self.snakes[agent_idx].dead:
                observations[agent_id] = obs  # Return empty observation for dead snakes
                continue

            for i, snake in enumerate(self.snakes):
                if snake.dead:
                    continue  # Skip dead snakes
                    
                for j, segment in enumerate(snake.body):
                    y, x = segment
                    if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                        if i == agent_idx:
                            obs[y, x] = 2 if j == 0 else 1  # Own head/body
                        else:
                            obs[y, x] = -2 if j == 0 else -1  # Other snakes

            # Apple is same for everyone
            if self.apple is not None:
                apple_y, apple_x = self.apple
                if 0 <= apple_y < self.grid_size[0] and 0 <= apple_x < self.grid_size[1]:
                    obs[apple_y, apple_x] = 3

            observations[agent_id] = obs

        return observations
    
    def _get_info(self) -> Dict[str, Dict]:
        # Initialize info dict for each agent
        infos = {}
        
        # Count living snakes
        living_snakes = sum(1 for snake in self.snakes if not snake.dead)
        snake_lengths = [len(snake.body) for snake in self.snakes]
        
        for agent_id in self.agents:
            agent_idx = self.agent_name_mapping[agent_id]
            infos[agent_id] = {
                "_num_agents": self._num_agents,
                "living_snakes": living_snakes,
                "snake_lengths": snake_lengths,
                "dead_snakes": [i for i, snake in enumerate(self.snakes) if snake.dead],
                "is_dead": self.snakes[agent_idx].dead
            }
            
        return infos
    
    def step(self, actions: Dict[str, int]):
        rewards = {agent: self._rewards["idle"] for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Process actions only for active agents
        agent_actions = [None] * self._num_agents
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                agent_idx = self.agent_name_mapping[agent_id]
                agent_actions[agent_idx] = action
        
        new_heads = []
        head_positions = {}
        
        # 1. Move all living snakes
        for idx, snake in enumerate(self.snakes):
            if snake.dead:
                agent_id = self.possible_agents[idx]
                if agent_id in self.agents:
                    terminations[agent_id] = True
                continue
            
            action = agent_actions[idx]
            if action is None:
                continue  # Skip if no action provided
                
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
                
            agent_id = self.possible_agents[idx]
            if self.apple is not None and np.array_equal(snake.body[0], self.apple):
                if agent_id in rewards:
                    rewards[agent_id] = self._rewards["win"]
                tail = snake.body[-1]
                snake.body = np.vstack([snake.body, tail[None, :]])  # Grow
                self._spawn_apple()

        # 3. Get updated occupied positions (includes bodies and walls)
        occupied_positions = self._get_occupied_postions()

        # 4. Collision Detection
        for idx, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            agent_id = self.possible_agents[idx]
            head = tuple(snake.body[0])

            # Head-on collision
            if head in head_positions and len(head_positions[head]) > 1:
                if agent_id in rewards:
                    rewards[agent_id] = self._rewards["lose"]
                    terminations[agent_id] = True
                snake.dead = True

            # Collision with wall or another snake's body
            elif head in occupied_positions:
                # Check if it's not colliding with its own body (first segment is head)
                is_self_collision = False
                for segment in snake.body[1:]:
                    if np.array_equal(head, tuple(segment)):
                        is_self_collision = True
                        break
                        
                if not is_self_collision:
                    if agent_id in rewards:
                        rewards[agent_id] = self._rewards["lose"]
                        terminations[agent_id] = True
                    snake.dead = True
        
        # Update agents list - remove terminated agents
        self.agents = [agent_id for agent_id in self.agents if not terminations[agent_id]]
        
        observations = self._get_obs()
        infos = self._get_info()
        
        # Check if game is done (all snakes dead or only one left)
        living_snakes = sum(1 for snake in self.snakes if not snake.dead)
        if living_snakes <= 1:
            # Game is over when there's one or fewer snakes alive
            for agent_id in self.agents:
                terminations[agent_id] = True
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            return

        grid = np.zeros((*self.grid_size, 3), dtype=np.uint8)
        
        # Draw the snakes
        for i, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            # Create a unique color for each snake
            color = [(i * 50) % 255, ((i * 100) + 50) % 255, ((i * 150) + 100) % 255]
            
            # Draw the body segments
            for j, segment in enumerate(snake.body):
                y, x = segment
                if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                    # Darken the color for body segments
                    if j == 0:  # Head
                        grid[y, x] = color
                    else:  # Body
                        grid[y, x] = [max(0, c - 50) for c in color]
        
        # Draw the apple
        if self.apple is not None:
            apple_y, apple_x = self.apple
            if 0 <= apple_y < self.grid_size[0] and 0 <= apple_x < self.grid_size[1]:
                grid[apple_y, apple_x] = [255, 0, 0]  # Red for apple
        
        # Scale the grid for better visualization
        cell_size = 20
        scaled_grid = np.zeros((self.grid_size[0] * cell_size, self.grid_size[1] * cell_size, 3), dtype=np.uint8)
        
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                scaled_grid[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size] = grid[y, x]
        
        if self.render_mode == "human":
            try:
                import pygame
                
                if not hasattr(self, 'screen'):
                    pygame.init()
                    self.screen = pygame.display.set_mode((self.grid_size[1] * cell_size, self.grid_size[0] * cell_size))
                    pygame.display.set_caption("Slitherin")
                    self.clock = pygame.time.Clock()
                
                # Clear the screen
                self.screen.fill((0, 0, 0))
                
                # Draw the grid
                for y in range(self.grid_size[0]):
                    for x in range(self.grid_size[1]):
                        pygame.draw.rect(
                            self.screen,
                            grid[y, x],
                            pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                        )
                
                # Update the display
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
                
            except ImportError:
                print("pygame not installed, using text mode")
                self._render_text()
                
        elif self.render_mode == "rgb_array":
            return scaled_grid

    def _render_text(self):
        """Render the environment as text in terminal."""
        grid = np.zeros(self.grid_size, dtype=str)
        grid.fill(' ')
        
        # Draw the snakes
        for i, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            for j, segment in enumerate(snake.body):
                y, x = segment
                if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                    if j == 0:  # Head
                        grid[y, x] = str(i)
                    else:  # Body
                        grid[y, x] = '●'
        
        # Draw the apple
        if self.apple is not None:
            apple_y, apple_x = self.apple
            if 0 <= apple_y < self.grid_size[0] and 0 <= apple_x < self.grid_size[1]:
                grid[apple_y, apple_x] = 'A'
        
        # Print the grid
        print('-' * (self.grid_size[1] + 2))
        for row in grid:
            print('|' + ''.join(row) + '|')
        print('-' * (self.grid_size[1] + 2))
        
    def close(self):
        """Close the environment."""
        if hasattr(self, 'screen'):
            import pygame
            pygame.quit()