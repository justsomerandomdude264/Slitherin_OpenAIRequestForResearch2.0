from typing import Optional
import numpy as np

class Snake():
    def __init__(
            self, 
            start_pos: tuple[int, int] = (0, 0),
            starting_direction: Optional[int] = 0,
            color: Optional[int] = 0
        ):
        """
        Initialize a snake with starting position and direction
        
        Parameters:
        - start_pos: tuple (y, x) coordinates
        - starting_direction: 0=right, 1=up, 2=left, 3=down
        - color: color index for rendering
        """
        self.body = np.array([start_pos], dtype=np.int32)
        self.direction = starting_direction
        self.color = color
        self.dead = False
        
        # Direction vectors: right(0,1), up(-1,0), left(0,-1), down(1,0)
        self._dir_vectors = np.array([
            [0, 1],   # right
            [-1, 0],  # up
            [0, -1],  # left
            [1, 0]    # down
        ])
    
    def move(self, new_direction):
        """
        Move the snake in the given direction
        """
        # Update direction
        self.direction = new_direction
        
        # Get movement vector based on direction
        move_vector = self._dir_vectors[self.direction]
        
        # Calculate new head position
        new_head = self.body[0] + move_vector
        
        # Update body (remove tail, add new head)
        self.body = np.vstack([new_head[None, :], self.body[:-1]])
        
    def grow(self) -> None:
        """
        Grow the snake by adding a segment at the tail
        """
        # Duplicate the last segment to grow
        tail = self.body[-1]
        self.body = np.vstack([self.body, tail[None, :]])

    def add_score(self) -> None:
        """
        Add 1 to the score
        """
        self.score += 1