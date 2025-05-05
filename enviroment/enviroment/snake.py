from typing import Optional
import numpy as np

class Snake():
    def __init__(
            self, 
            start_pos: Optional[tuple[int, int]] = (0, 0),
            starting_direction: Optional[int] = 0,
            color: Optional[int] = 0
        ):

        # Define score
        self.score = 0

        # Define the blocks, the first is the head of the snake.
        self.blocks = np.array([start_pos])

        # Define the direction of the snake
        self.direction = starting_direction

        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        # Map color int to respective color (default to red)
        color_map = {
            0: "red",
            1: "blue",
            2: "yellow",
            3: "green",
            4: "pink",
            5: "purple",
        }
        self.color = color_map.get(color, "red") 

    def reset(
            self, 
            start_pos: Optional[tuple[int, int]] = (0, 0),
            starting_direction: Optional[int] = 0
        ) -> None:
        # Define the blocks, the first is the head of the snake
        self.blocks = np.array([start_pos])
        # Define the direction of the snake
        self.direction = starting_direction

    def move(
            self,
            direction: Optional[int] = None,
        ) -> None:
         # If no direction is given, continue in current direction
        if direction is not None:
            self.direction = direction
        
        move_vector = self._action_to_direction[self.direction]
        new_head = self.blocks[0] + move_vector

        # Shift body: add new head and remove the tail
        self.blocks = np.insert(self.blocks, 0, [new_head], axis=0)
        self.blocks = np.delete(self.blocks, -1, axis=0)

    def add_score(
            self
        ) -> None:
        self.score += 1