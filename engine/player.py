import numpy as np
from .map import Map

class Player:
    MAX_BOMB_RADIUS = 5
    MAX_BOMB_CAPACITY = 5
    STOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    PLACE_BOMB = 5
    
    def __init__(self, player_id, row, col):
        self.id = player_id
        self.x = row
        self.y = col
        self.alive = True
        self.bombs_left = 1
        self.bomb_radius_bonus = 0
        self.stats = {'kills': 0, 'boxes': 0, 'items': 0, 'bombs': 0}
    
    def move(self, dx, dy, grid, players, bombs):
        if not self.alive:
            return
        
        new_x = self.x + dx
        new_y = self.y + dy
    
        if not (0 < new_x < grid.shape[0] - 1 and 0 < new_y < grid.shape[1] - 1):
            return
        if grid[new_x, new_y] in [Map.WALL, Map.BOX]:
            return
        if any(b.x == new_x and b.y == new_y for b in bombs):
            return
        
        # players can overlap
        self.x = new_x
        self.y = new_y