import numpy as np
from .map import Map
from .bomb import Bomb
from .player import Player
class BomberEnv:
    # N_ACTIONS = 6 # 0: STOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: PLACE_BOMB
    
    def __init__(self, width=13, height=13, max_steps=500, seed=None):
        self.width = width 
        self.height = height
        self.max_steps = max_steps
        self.seed_val = seed
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)
        
    def reset(self, seed=None, options=None):
        # reset() gets different seed when simulating multiple games
        if seed is not None:
            self.seed_val = seed
            self.rng = np.random.default_rng(seed)

        self.map = Map(self.width, self.height, seed=self.seed_val)
        self.players = [
            Player(0, 1, 1),
            Player(1, self.height - 2, self.width - 2),
            Player(2, 1, self.width - 2),
            Player(3, self.height - 2, 1)
        ]
        
        self.bombs = []
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        # Full observability
        # Map: 0: grass, 1: wall, 2: box, 3: item_radius, 4: item_capacity
        # Players: [x, y, alive, bombs_left, bomb_radius_bonus]
        # Bombs: [x, y, timer, owner_id]
        return {
            "map": self.map.grid.copy(),
            "players": np.array([[p.x, p.y, p.alive, p.bombs_left, p.bomb_radius_bonus] for p in self.players], dtype=np.int8),
            "bombs": np.array([[b.x, b.y, b.timer, b.owner_id] for b in self.bombs], dtype=np.int8)
        }
        
    # Resolution: actions first (movement, bomb placement) -> bomb explosion -> spawn items -> check termination
    def step(self, actions):
        self.current_step += 1
        pending_bombs = {}
        
        for player_id, action in enumerate(actions):
            player = self.players[player_id]
            if player.alive == False:
                continue
            
            dx, dy = 0, 0
            if action == Player.LEFT: dx = -1
            elif action == Player.RIGHT: dx = 1
            elif action == Player.UP: dy = -1
            elif action == Player.DOWN: dy = 1
            elif action == Player.PLACE_BOMB:
                if player.bombs_left <= 0:
                    continue
                # if there is a bomb already (from previous step) -> cannot place, but same step bomb placement is allowed (pending bombs)
                if any(b.x == player.x and b.y == player.y for b in self.bombs):
                    continue
                # if multiple players place bomb in the same cell, only place the bomb with largest radius
                radius = 1 + player.bomb_radius_bonus
                pos = (player.x, player.y)
                if pos not in pending_bombs or radius > pending_bombs[pos][1]:
                    pending_bombs[pos] = (player_id, radius)
            
            if dx != 0 or dy != 0:
                player.move(dx, dy, self.map.grid, self.players, self.bombs) # move first
                
        # Resolve item collections after all movements
        tile_to_players = {}
        for p in self.players:
            if p.alive:
                pos = (p.x, p.y)
                if pos not in tile_to_players:
                    tile_to_players[pos] = []
                tile_to_players[pos].append(p)
                
        for (x, y), occupants in tile_to_players.items():
            cell = self.map.grid[x, y]
            if cell in [Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
                if len(occupants) == 1:
                    p = occupants[0]
                    if cell == Map.ITEM_RADIUS:
                        p.bomb_radius_bonus = min(p.bomb_radius_bonus + 1, Player.MAX_BOMB_RADIUS - 1)
                        p.stats['items'] += 1
                    elif cell == Map.ITEM_CAPACITY:
                        p.bombs_left = min(p.bombs_left + 1, Player.MAX_BOMB_CAPACITY)
                        p.stats['items'] += 1
                # Remove the item whether collected (1 occupant) or destroyed (>1 occupant)
                self.map.grid[x, y] = Map.GRASS

        for (bx, by), (owner_id, radius) in pending_bombs.items():
            self.bombs.append(Bomb(bx, by, owner_id, radius=radius)) # then place bombs
            self.players[owner_id].bombs_left -= 1
            self.players[owner_id].stats['bombs'] += 1
            
        exploded_this_step = []
        # tick all bombs
        for bomb in self.bombs:
            if bomb.step():
                exploded_this_step.append(bomb)
        # chain reaction
        idx = 0
        while idx < len(exploded_this_step):
            bomb = exploded_this_step[idx]
            idx += 1
            
            explosion_tiles = self._get_explosion_tiles(bomb)
            
            for other_bomb in self.bombs:
                if other_bomb.exploded:
                    continue
                if (other_bomb.x, other_bomb.y) in explosion_tiles and other_bomb not in exploded_this_step:
                    other_bomb.exploded = True
                    exploded_this_step.append(other_bomb)
        
        if exploded_this_step:
            affected_tiles_map = {}
            for bomb in exploded_this_step:
                for tx, ty in self._get_explosion_tiles(bomb):
                    if (tx, ty) not in affected_tiles_map:
                        affected_tiles_map[(tx, ty)] = set()
                    affected_tiles_map[(tx, ty)].add(bomb.owner_id)
                self.players[bomb.owner_id].bombs_left += 1
            
            self._apply_explosions(affected_tiles_map)
            self.bombs = [b for b in self.bombs if not b.exploded]
        self._spawn_random_items()
        
        terminated = sum(p.alive for p in self.players) <= 1
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), terminated, truncated
    
    def _get_explosion_tiles(self, bomb):
        tiles = {(bomb.x, bomb.y)}
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # bomb.radius already includes owner's bonus at placement time
        for dx, dy in directions:
            for r in range(1, bomb.radius + 1):
                tx, ty = bomb.x + dx * r, bomb.y + dy * r
                if 0 <= tx < self.height and 0 <= ty < self.width:
                    if self.map.grid[tx, ty] == Map.WALL:
                        break
                    tiles.add((tx, ty))
                    if self.map.grid[tx, ty] == Map.BOX:
                        break
                    # exlposion goes through players
                else:
                    break
        
        return tiles

    def _apply_explosions(self, affected_tiles_map):
        for (tx, ty), owner_ids in affected_tiles_map.items():
            for p in self.players:
                if p.x == tx and p.y == ty and p.alive:
                    p.alive = False
                    for oid in owner_ids:
                        if oid != p.id:
                            self.players[oid].stats['kills'] += 1
            if self.map.grid[tx, ty] == Map.BOX:
                self.map.grid[tx, ty] = Map.GRASS
                for oid in owner_ids:
                    self.players[oid].stats['boxes'] += 1
                rand = self.rng.random()
                if rand < 0.3:
                    self.map.grid[tx, ty] = Map.ITEM_RADIUS
                elif rand < 0.6:
                    self.map.grid[tx, ty] = Map.ITEM_CAPACITY
            # can destroy items
            elif self.map.grid[tx, ty] in [Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
                self.map.grid[tx, ty] = Map.GRASS
        
    def _spawn_random_items(self, spawn_prob=0.0003):
        for x in range(self.height):
            for y in range(self.width):
                if self.map.grid[x, y] != Map.GRASS:
                    continue
                if any(p.x == x and p.y == y and p.alive for p in self.players):
                    continue
                if self.rng.random() < spawn_prob * self.current_step / 165:
                    if self.rng.random() < 0.5:
                        self.map.grid[x, y] = Map.ITEM_RADIUS
                    else:
                        self.map.grid[x, y] = Map.ITEM_CAPACITY