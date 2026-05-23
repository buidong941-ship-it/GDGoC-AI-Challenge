import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
# Add parent directory to sys.path if not already present
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine import Map

# Defaults when bomb rows omit timer/owner (engine always sends 4-tuple)
_DEFAULT_BOMB_TIMER = 7
_DEFAULT_BOMB_OWNER = 0


def _parse_bomb_row(b):
    """Return (bx, by, timer, owner_id); timer/owner default if only (x, y) is given."""
    arr = np.asarray(b, dtype=np.float64).ravel()
    if arr.size < 2:
        return None
    bx, by = int(arr[0]), int(arr[1])
    timer = int(arr[2]) if arr.size > 2 else _DEFAULT_BOMB_TIMER
    owner_id = int(arr[3]) if arr.size > 3 else _DEFAULT_BOMB_OWNER
    return bx, by, timer, owner_id


REWARD_DICT = {
    "win": 2.0,
    "enemy_death": 1.0,
    "agent_death": -2.0,
    "standing_still": -0.01,
    "time_penalty": -0.005,
    "plant_near_box": 0.05,
    "item_collection": 0.1,
    "danger_evasion": 0.12,
    "danger_enter": -0.06,
    "own_blast_loiter": -0.04,
    "approach_enemy": 0.02,
}


def _bomb_radius_from_obs(players, owner_id):
    return 1 + int(players[int(owner_id)][4])


def _explosion_tiles_for_bomb(grid, bx, by, radius):
    """Same cross-shaped blast rules as BomberEnv._get_explosion_tiles."""
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == Map.WALL:
                break
            tiles.add((tx, ty))
            if cell == Map.BOX:
                break
    return tiles


def _blast_status_at(obs, x, y):
    """
    Returns (in_blast: bool, min_timer: int|None) for active bombs in obs.
    min_timer is the smallest timer among bombs whose blast includes (x, y).
    """
    bombs = obs["bombs"]
    if bombs is None:
        return False, None
    arr = np.asarray(bombs)
    if arr.size == 0:
        return False, None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ix, iy = int(x), int(y)
    players = obs["players"]
    grid = obs["map"]
    in_blast = False
    min_timer = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        radius = _bomb_radius_from_obs(players, owner_id)
        tiles = _explosion_tiles_for_bomb(grid, bx, by, radius)
        if (ix, iy) in tiles:
            in_blast = True
            t = int(timer)
            min_timer = t if min_timer is None else min(min_timer, t)
    return in_blast, min_timer


def _any_bombs(obs):
    b = obs["bombs"]
    if b is None:
        return False
    return np.asarray(b).size > 0


def _enemy_alive_count(players, agent_id):
    """BomberEnv uses a (N, 5) ndarray; unit tests may use dict keyed by player id."""
    if isinstance(players, dict):
        return sum(
            1 for pid, p in players.items()
            if pid != agent_id and int(p[2]) == 1
        )
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n = arr.shape[0]
    return sum(
        1 for pid in range(n)
        if pid != agent_id and int(arr[pid][2]) == 1
    )


def _manhattan_to_nearest_alive_enemy(players, agent_id, x, y):
    """None if there is no other alive player."""
    best = None
    ix, iy = int(x), int(y)
    if isinstance(players, dict):
        for pid, p in players.items():
            if pid == agent_id or int(p[2]) != 1:
                continue
            d = abs(ix - int(p[0])) + abs(iy - int(p[1]))
            best = d if best is None else min(best, d)
        return best
    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for pid in range(arr.shape[0]):
        if pid == agent_id or int(arr[pid][2]) != 1:
            continue
        d = abs(ix - int(arr[pid][0])) + abs(iy - int(arr[pid][1]))
        best = d if best is None else min(best, d)
    return best


def _in_own_predicted_blast(obs, agent_id, x, y):
    return _min_own_blast_timer_at(obs, agent_id, x, y) is not None


def _min_own_blast_timer_at(obs, agent_id, x, y):
    """Smallest tick countdown among this agent's bombs whose blast includes (x, y)."""
    bombs = obs["bombs"]
    if bombs is None:
        return None
    arr = np.asarray(bombs)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    players = obs["players"]
    grid = obs["map"]
    ix, iy = int(x), int(y)
    aid = int(agent_id)
    best = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        if int(owner_id) != aid:
            continue
        radius = _bomb_radius_from_obs(players, owner_id)
        tiles = _explosion_tiles_for_bomb(grid, bx, by, radius)
        if (ix, iy) in tiles:
            t = int(timer)
            best = t if best is None else min(best, t)
    return best


def compute_reward(prev_obs, curr_obs, agent_id):
    if prev_obs is None:
        return 0.0

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]
    
    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    reward = 0.0
    
    # 1. WIN / LOSS CONDITIONS
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARD_DICT["agent_death"])
    
    prev_enemies_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemies_alive = _enemy_alive_count(curr_players, agent_id)
    
    if curr_enemies_alive < prev_enemies_alive:
        reward += REWARD_DICT["enemy_death"] * (prev_enemies_alive - curr_enemies_alive)
    if curr_enemies_alive == 0 and prev_enemies_alive > 0:
        reward += REWARD_DICT["win"]

    # 2. MOVEMENT & TIME PENALTIES
    prev_x, prev_y = prev_players[agent_id][0], prev_players[agent_id][1]
    curr_x, curr_y = curr_players[agent_id][0], curr_players[agent_id][1]
    
    if prev_x == curr_x and prev_y == curr_y:
        reward += REWARD_DICT["standing_still"]
    else:
        reward -= REWARD_DICT["standing_still"] # Moving still incurs a small time penalty to encourage efficiency
    
    reward += REWARD_DICT["time_penalty"]

    # 2b. DANGER EVASION — reward leaving predicted blast; penalize walking into it
    if _any_bombs(prev_obs) or _any_bombs(curr_obs):
        prev_in_blast, prev_timer = _blast_status_at(prev_obs, prev_x, prev_y)
        curr_in_blast, _ = _blast_status_at(curr_obs, curr_x, curr_y)
        if prev_in_blast and not curr_in_blast:
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 3) else 1.0
            reward += REWARD_DICT["danger_evasion"] * urgency
        elif (
            not prev_in_blast
            and curr_in_blast
            and (prev_x != curr_x or prev_y != curr_y)
        ):
            # Only when stepping into blast; standing still (e.g. planting on own tile) is excluded
            reward += REWARD_DICT["danger_enter"]

    # Standing in your own blast: penalize more as the fuse runs down (clearer than flat -0.04).
    mt_own = _min_own_blast_timer_at(curr_obs, agent_id, curr_x, curr_y)
    if curr_alive == 1 and mt_own is not None:
        urgency = max(1, 8 - int(mt_own))
        reward += REWARD_DICT["own_blast_loiter"] * float(urgency)

    if (
        curr_alive == 1
        and prev_enemies_alive > 0
        and curr_enemies_alive > 0
    ):
        prev_d = _manhattan_to_nearest_alive_enemy(prev_players, agent_id, prev_x, prev_y)
        curr_d = _manhattan_to_nearest_alive_enemy(curr_players, agent_id, curr_x, curr_y)
        if prev_d is not None and curr_d is not None:
            reward += REWARD_DICT["approach_enemy"] * (prev_d - curr_d)

    # 3. ITEM COLLECTION
    # Based on your legend: 3 is item_radius, 4 is item_capacity
    stepped_on_tile = prev_obs["map"][curr_x, curr_y]
    if stepped_on_tile in [3, 4]: 
        reward += REWARD_DICT["item_collection"]
    else:
        # Fallback check just in case items spawn under players or map updates differently
        prev_radius_bonus = int(prev_players[agent_id][4])
        curr_radius_bonus = int(curr_players[agent_id][4])
        if curr_radius_bonus > prev_radius_bonus:
             reward += REWARD_DICT["item_collection"]

    # 4. REWARD SHAPING: Box Destruction Proxy
    prev_bombs_left = int(prev_players[agent_id][3])
    curr_bombs_left = int(curr_players[agent_id][3])
    
    if curr_bombs_left < prev_bombs_left:
        # Check immediate adjacent tiles (up, down, left, right)
        adjacent_tiles = [
            prev_obs["map"][max(0, curr_x-1), curr_y],
            prev_obs["map"][min(prev_obs["map"].shape[0]-1, curr_x+1), curr_y],
            prev_obs["map"][curr_x, max(0, curr_y-1)],
            prev_obs["map"][curr_x, min(prev_obs["map"].shape[1]-1, curr_y+1)]
        ]
        
        # 2 is the integer for "box" based on your legend
        if 2 in adjacent_tiles:
            reward += REWARD_DICT["plant_near_box"]

    return float(reward)


class UnitTestReward:
    def agent_death(self):
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)}, # Changed True to 1 for alive flag
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 0, 1, 1)}, # Changed False to 0
            "bombs": []
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Agent Death Reward: {reward}")
        assert reward == REWARD_DICT["agent_death"], "Expected exactly the agent death penalty"
    
    def agent_standing_still(self):
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)},
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)},
            "bombs": []
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Agent Standing Still Reward: {reward}")
        expected = REWARD_DICT["standing_still"] + REWARD_DICT["time_penalty"]
        assert reward == expected, f"Expected {expected} for standing still"
    
    def agent_moving(self):
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)},
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 2, 1, 1, 1)}, # Player moved
            "bombs": []
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Agent Moving Reward: {reward}")
        # Moving avoids the standing still penalty, but still incurs the time penalty
        expected = -REWARD_DICT["time_penalty"] 
        assert reward == expected, f"Expected {expected} for just moving"
    
    def agent_plant_near_box(self):
        # Replaces `box_destruction`. We now test if the agent gets rewarded for 
        # dropping a bomb directly next to a box.
        prev_obs = {
            "map": np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]]), # Box at (0,1)
            "players": {0: (1, 1, 1, 1, 1)}, # Player at (1,1), adjacent to box
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 0, 1)}, # Ammo dropped from 1 to 0
            "bombs": [(1, 1)]
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Plant Near Box Reward: {reward}")
        expected = (
            REWARD_DICT["standing_still"]
            + REWARD_DICT["time_penalty"]
            + REWARD_DICT["plant_near_box"]
            + REWARD_DICT["own_blast_loiter"]
        )
        assert reward == expected, "Expected reward for planting near a box"
    
    def item_collection(self):
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 1, 1)}, # Radius is 1
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 2, 2)}, # Radius increased to 2
            "bombs": []
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Item Collection Reward: {reward}")
        expected = REWARD_DICT["standing_still"] + REWARD_DICT["time_penalty"] + REWARD_DICT["item_collection"]
        assert reward == expected, "Expected positive reward for item collection"
    
    def agent_place_bomb_no_box(self):
        # Renamed for clarity. Placing a bomb with NO boxes around should just be a normal turn.
        prev_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), # Wall nearby, no box
            "players": {0: (1, 1, 1, 1, 1)},
            "bombs": []
        }
        curr_obs = {
            "map": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "players": {0: (1, 1, 1, 0, 1)}, # Ammo dropped
            "bombs": [(1, 1)]
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Agent Place Bomb (No Box) Reward: {reward}")
        expected = (
            REWARD_DICT["standing_still"]
            + REWARD_DICT["time_penalty"]
            + REWARD_DICT["own_blast_loiter"]
        )
        assert reward == expected, "Expected standing/time + own-blast loiter for bomb on self"

    def danger_evasion_leave_blast(self):
        # Bomb at (2,2) radius 1 blasts (2,3); agent steps from (2,3) to (3,3).
        m = np.zeros((5, 5), dtype=np.int8)
        prev_obs = {
            "map": m,
            "players": {0: (2, 3, 1, 1, 0), 1: (0, 0, 1, 1, 0)},
            "bombs": np.array([[2, 2, 5, 1]], dtype=np.int8),
        }
        curr_obs = {
            "map": m,
            "players": {0: (3, 3, 1, 1, 0), 1: (0, 0, 1, 1, 0)},
            "bombs": np.array([[2, 2, 4, 1]], dtype=np.int8),
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Danger Evasion Reward: {reward}")
        # Enemy at (0,0): Manhattan 5 -> 6 (one step away); approach term -1 * scale
        approach = REWARD_DICT["approach_enemy"] * (5 - 6)
        expected = (
            -REWARD_DICT["standing_still"]
            + REWARD_DICT["time_penalty"]
            + REWARD_DICT["danger_evasion"]
            + approach
        )
        assert abs(reward - expected) < 1e-6, f"Expected ~{expected}, got {reward}"

    def danger_evasion_urgent_timer(self):
        m = np.zeros((5, 5), dtype=np.int8)
        prev_obs = {
            "map": m,
            "players": {0: (2, 3, 1, 1, 0), 1: (0, 0, 1, 1, 0)},
            "bombs": np.array([[2, 2, 2, 1]], dtype=np.int8),
        }
        curr_obs = {
            "map": m,
            "players": {0: (3, 3, 1, 1, 0), 1: (0, 0, 1, 1, 0)},
            "bombs": np.array([[2, 2, 1, 1]], dtype=np.int8),
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Danger Evasion (urgent) Reward: {reward}")
        approach = REWARD_DICT["approach_enemy"] * (5 - 6)
        expected = (
            -REWARD_DICT["standing_still"]
            + REWARD_DICT["time_penalty"]
            + REWARD_DICT["danger_evasion"] * 1.5
            + approach
        )
        assert abs(reward - expected) < 1e-6, f"Expected ~{expected}, got {reward}"

    def approach_enemy_closer(self):
        prev_obs = {
            "map": np.zeros((3, 3), dtype=np.int8),
            "players": {0: (0, 0, 1, 1, 0), 1: (2, 2, 1, 1, 0)},
            "bombs": [],
        }
        curr_obs = {
            "map": np.zeros((3, 3), dtype=np.int8),
            "players": {0: (1, 1, 1, 1, 0), 1: (2, 2, 1, 1, 0)},
            "bombs": [],
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Approach Enemy Reward: {reward}")
        prev_d, curr_d = 4, 2
        expected = (
            -REWARD_DICT["standing_still"]
            + REWARD_DICT["time_penalty"]
            + REWARD_DICT["approach_enemy"] * (prev_d - curr_d)
        )
        assert abs(reward - expected) < 1e-6, f"Expected ~{expected}, got {reward}"

    def danger_enter_blast(self):
        m = np.zeros((5, 5), dtype=np.int8)
        prev_obs = {
            "map": m,
            "players": {0: (3, 3, 1, 1, 0), 1: (0, 0, 1, 1, 0)},
            "bombs": np.array([[2, 2, 5, 1]], dtype=np.int8),
        }
        curr_obs = {
            "map": m,
            "players": {0: (2, 3, 1, 1, 0), 1: (0, 0, 1, 1, 0)},
            "bombs": np.array([[2, 2, 4, 1]], dtype=np.int8),
        }
        reward = compute_reward(prev_obs, curr_obs, agent_id=0)
        print(f"Danger Enter Reward: {reward}")
        approach = REWARD_DICT["approach_enemy"] * (6 - 5)
        expected = (
            -REWARD_DICT["standing_still"]
            + REWARD_DICT["time_penalty"]
            + REWARD_DICT["danger_enter"]
            + approach
        )
        assert abs(reward - expected) < 1e-6, f"Expected ~{expected}, got {reward}"

    def run_all_tests(self):
        self.agent_death()
        self.agent_standing_still()
        self.agent_moving()
        self.agent_plant_near_box()
        self.item_collection()
        self.agent_place_bomb_no_box()
        self.approach_enemy_closer()
        self.danger_evasion_leave_blast()
        self.danger_evasion_urgent_timer()
        self.danger_enter_blast()
        print("All reward tests passed!")

if __name__ == "__main__":
    tester = UnitTestReward()
    tester.run_all_tests()