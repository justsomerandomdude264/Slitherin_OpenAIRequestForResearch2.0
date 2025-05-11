from typing import Optional, TypedDict, Dict, List, Tuple, Any
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
        self.score = 0
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

    def add_score(self):
        """Add one to the snake's score, used when collides with an apple."""
        self.score += 1


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
        
        # Store number of agents as an instance attribute (not a property)
        self._num_agents = num_agents
        
        # PettingZoo requires agent ids to be strings
        self.possible_agents = [f"snake_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()
        
        # Define reward
        self.rewards = rewards
        
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

    @property
    def num_agents(self):
        """Getter for number of agents"""
        return self._num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _ensure_valid_starting_positions(self):
        """
        Make sure snakes start at valid positions and don't overlap.
        """
        occupied_positions = set()
        
        for snake in self.snakes:
            if not snake.dead:
                # Trying up to 10 times to find a non-overlapping position
                for _ in range(10):
                    new_pos = (random.randint(1, self.grid_size[0] - 2), 
                              random.randint(1, self.grid_size[1] - 2))
                    
                    if new_pos not in occupied_positions:
                        snake.body = np.array([new_pos])
                        occupied_positions.add(new_pos)
                        break

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset agents list to all possible agents
        self.agents = self.possible_agents.copy()
        
        # Reset the snakes with positions not too close to the edge
        self.snakes = [
            Snake(
                # Use positions away from the edges to avoid immediate wall collisions
                start_pos=(random.randint(1, self.grid_size[0] - 2), 
                          random.randint(1, self.grid_size[1] - 2)),
                starting_direction=random.randint(0, 3),  # 0=right, 1=up, 2=left, 3=down
                color=random.randint(0, 5)
            )
            for _ in range(self._num_agents)
        ]
        
        # Ensure valid starting positions
        self._ensure_valid_starting_positions()
        
        # Reset apple position
        self._spawn_apple()
        
        # Get observations and info
        observations = self._get_obs()
        infos = self._get_info()
        
        return observations, infos

    def _get_occupied_postions(self):
        """Returns positions occupied by snake bodies (excluding heads) and walls."""
        _occupied_positions = set()

        # Snake bodies (exclude heads to avoid self-collisions on first move)
        for _snake in self.snakes:
            if not _snake.dead and len(_snake.body) > 1:  # Only add body segments, not heads
                for _segment in _snake.body[1:]:  # Skip the head at index 0
                    _occupied_positions.add(tuple(_segment))

        # Walls - use actual grid size boundaries
        for x in range(self.grid_size[0]):
            _occupied_positions.add((x, -1))  # Bottom wall
            _occupied_positions.add((x, self.grid_size[1]))  # Top wall
        
        for y in range(self.grid_size[1]):
            _occupied_positions.add((-1, y))  # Left wall
            _occupied_positions.add((self.grid_size[0], y))  # Right wall

        return _occupied_positions
    
    def _get_all_occupied_positions(self):
        """Get all occupied positions including snake heads."""
        positions = self._get_occupied_postions()
        
        # Add snake heads
        for snake in self.snakes:
            if not snake.dead:
                positions.add(tuple(snake.body[0]))
                
        return positions
    
    def _spawn_apple(self):
        # Get all the available positions by removing the occupied ones
        occupied = self._get_all_occupied_positions()
        
        _all_available_positions = [
            (x, y)
            for x in range(self.grid_size[0])
            for y in range(self.grid_size[1])
            if (x, y) not in occupied
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
                "num_agents": self._num_agents,
                "living_snakes": living_snakes,
                "snake_lengths": snake_lengths,
                "snake_score": self.snakes[agent_idx].score,
                "dead_snakes": [i for i, snake in enumerate(self.snakes) if snake.dead],
                "is_dead": self.snakes[agent_idx].dead
            }
            
        return infos
    
    def _check_collision_with_wall(self, head_position):
        """Check if head position is outside the grid."""
        y, x = head_position
        return y < 0 or y >= self.grid_size[0] or x < 0 or x >= self.grid_size[1]
    
    def _check_collision_with_body(self, head_position, snake_index):
        """Check if head position collides with any snake body (including own)."""
        for i, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            # For own snake, check collision with body (not head)
            if i == snake_index:
                for segment in snake.body[1:]:  # Skip own head
                    if np.array_equal(head_position, segment):
                        return True
            # For other snakes, check collision with entire body including head
            else:
                for segment in snake.body:
                    if np.array_equal(head_position, segment):
                        return True
        return False
    
    def step(self, actions: Dict[str, int]):
        rewards = {agent: self.rewards["idle"] for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Process actions only for active agents
        agent_actions = [None] * self._num_agents
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                agent_idx = self.agent_name_mapping[agent_id]
                agent_actions[agent_idx] = action
        
        # Store new head positions for collision checking
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
                
            # Prevent reversing into your own body (only if length > 1)
            if len(snake.body) > 1 and abs(action - snake.direction) == 2:
                action = snake.direction
                
            # Store the old head before moving
            #old_head = tuple(snake.body[0])
            
            # Move the snake
            snake.move(action)
            new_head = tuple(snake.body[0])
            
            # Add to head positions for collision detection
            new_heads.append(new_head)
            head_positions.setdefault(new_head, []).append(idx)
        
        # 2. Check collisions with walls first
        for idx, snake in enumerate(self.snakes):
            if snake.dead:
                continue
            
            agent_id = self.possible_agents[idx]
            head_position = tuple(snake.body[0])
            
            # Check wall collision
            if self._check_collision_with_wall(head_position):
                if agent_id in rewards:
                    rewards[agent_id] = self.rewards["lose"]
                    terminations[agent_id] = True
                snake.dead = True
        
        # 3. Check snake-to-snake collisions (head-on and head-to-body)
        for idx, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            agent_id = self.possible_agents[idx]
            head_position = tuple(snake.body[0])
            
            # Head-on collision with another snake
            if head_position in head_positions and len(head_positions[head_position]) > 1:
                if agent_id in rewards:
                    rewards[agent_id] = self.rewards["lose"]
                    terminations[agent_id] = True
                snake.dead = True
                continue
            
            # Collision with any snake body
            if self._check_collision_with_body(head_position, idx):
                if agent_id in rewards:
                    rewards[agent_id] = self.rewards["lose"]
                    terminations[agent_id] = True
                snake.dead = True

        # 4. Grow if on apple - only for snakes that didn't die in this step
        for idx, snake in enumerate(self.snakes):
            if snake.dead:
                continue
                
            agent_id = self.possible_agents[idx]
            if self.apple is not None and np.array_equal(snake.body[0], self.apple):
                if agent_id in rewards:
                    rewards[agent_id] = self.rewards["win"]
                    snake.add_score()
                
                # Grow the snake
                if len(snake.body) > 0:
                    tail = snake.body[-1]
                    snake.body = np.vstack([snake.body, tail[None, :]])
                
                # Spawn a new apple
                self._spawn_apple()
        
        # Update agents list - remove terminated agents
        self.agents = [agent_id for agent_id in self.agents if not terminations[agent_id]]
        
        # Get observations and info
        observations = self._get_obs()
        infos = self._get_info()
        
        # Check if game is done (all snakes dead or only one left)
        living_snakes = sum(1 for snake in self.snakes if not snake.dead)
        if living_snakes <= 1 and self._num_agents > 1:
            # Game is over when there's only one snake alive in multiplayer
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