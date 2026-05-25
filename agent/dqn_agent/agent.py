from pathlib import Path
from collections import deque
import numpy as np
from tqdm import tqdm
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim


# Constants from engine
class Map:
    GRASS = 0
    WALL = 1
    BOX = 2
    ITEM_RADIUS = 3
    ITEM_CAPACITY = 4
    BOMB = 5

class Player:
    MAX_BOMB_RADIUS = 5
    MAX_BOMB_CAPACITY = 5

BOMB_MAX_TIMER = 7

class ReplayBuffer:
    """Pre-allocated numpy circular buffer — sample() is pure array indexing, no Python objects."""
    def __init__(self, capacity: int, map_shape, aux_dim: int, num_actions: int = 6):
        self.capacity  = capacity
        self.pos       = 0
        self.size      = 0
        self.map_shape = tuple(map_shape)
        self.aux_dim   = int(aux_dim)
        self.num_actions = int(num_actions)
        self.map_states      = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.aux_states      = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.next_map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.next_aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.action_masks      = np.ones((capacity, self.num_actions), dtype=np.float32)
        self.next_action_masks = np.ones((capacity, self.num_actions), dtype=np.float32)
        self.actions     = np.zeros(capacity,              dtype=np.int64)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.dones       = np.zeros(capacity,              dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, map_state, aux_state, action, reward, next_map_state, next_aux_state, done, action_mask=None, next_action_mask=None):
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.map_states[self.pos]      = map_state
        self.aux_states[self.pos]      = aux_state
        self.next_map_states[self.pos] = next_map_state
        self.next_aux_states[self.pos] = next_aux_state
        self.action_masks[self.pos] = self._normalise_mask(action_mask)
        self.next_action_masks[self.pos] = self._normalise_mask(next_action_mask)
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done

    def _normalise_mask(self, mask):
        if mask is None:
            return np.ones(self.num_actions, dtype=np.float32)
        arr = np.asarray(mask, dtype=np.float32)
        if arr.shape != (self.num_actions,):
            out = np.zeros(self.num_actions, dtype=np.float32)
            for action in mask:
                action = int(action)
                if 0 <= action < self.num_actions:
                    out[action] = 1.0
            arr = out
        if not np.any(arr > 0):
            arr[0] = 1.0
        return arr

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.map_states[idx],
            self.aux_states[idx],
            self.next_map_states[idx],
            self.next_aux_states[idx],
            self.action_masks[idx],
            self.next_action_masks[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )

class DQNModel(nn.Module):
    """
    Two-branch DQN:
      - Conv2D branch for spatial map/object channels
      - MLP branch for auxiliary scalar features
    """
    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, h, w = map_shape
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.map_encoder(dummy).reshape(1, -1).size(1)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, map_x, aux_x):
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        feat = torch.cat([map_feat, aux_feat], dim=1)
        return self.head(feat)

def encode_obs(obs, agent_ids):
    """
    Returns:
      map_feat: spatial tensor for Conv2D branch, shape (C, H, W)
      aux_feat: scalar tensor for auxiliary branch, shape (A,)

    agent_ids: int (user's player id) or list/tuple [user_id, opp_id].
    When a single int is given the enemy is inferred as the other
    player in a 2-player game (1 - user_id).
    """
    if obs is None:
        raise ValueError("obs should not be None")

    # Normalise agent_ids to (user_id, opp_id)
    user_id = int(agent_ids[0])
    opp_id  = int(agent_ids[1]) if len(agent_ids) > 1 else (1 - user_id)

    grid    = obs["map"]      # (H, W)
    players = obs["players"]  # (num_players, 5)
    bombs   = obs["bombs"]    # (N, 4), N may be 0
    H, W    = grid.shape

    # One-hot map: grass, wall, box, item_radius, item_capacity
    map_channels = []
    for v in [Map.GRASS, Map.WALL, Map.BOX, Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
        map_channels.append((grid == v).astype(np.float32))
    # Player position masks
    my_x, my_y, my_alive, my_bombs_left, my_radius_bonus = players[user_id]
    ox,   oy,   opp_alive, _,            _               = players[opp_id]
    my_pos  = np.zeros((H, W), dtype=np.float32)
    opp_pos = np.zeros((H, W), dtype=np.float32)
    if int(my_alive)  == 1:
        my_pos[int(my_x), int(my_y)] = 1.0
    if int(opp_alive) == 1:
        opp_pos[int(ox), int(oy)]    = 1.0

    # Bomb channels — bombs is a numpy array, not a list of Bomb objects
    bomb_timer = np.zeros((H, W), dtype=np.float32)
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by, timer, owner_id = b
        bx, by = int(bx), int(by)
        t = float(timer) / BOMB_MAX_TIMER  # normalise by default max timer
        bomb_timer[bx, by] = max(bomb_timer[bx, by], t)
        bomb_owned[bx, by] = 1.0 if int(owner_id) == user_id else 0.0

    scalar = np.array([
        float(my_bombs_left)   / Player.MAX_BOMB_CAPACITY,
        float(my_radius_bonus) / Player.MAX_BOMB_RADIUS,
        float(opp_alive),
    ], dtype=np.float32)

    map_feat = np.stack([
        *map_channels,          # 5 channels
        my_pos,                 # 1 channel
        opp_pos,                # 1 channel
        bomb_timer,             # 1 channel
        bomb_owned,             # 1 channel
    ], axis=0).astype(np.float32)  # (9, H, W)
    return map_feat, scalar

class TrainingAgent:
    """
    Agent class for DQN training and evaluation.
    Args:
        agent_id: int
        input_dim: int
        num_actions: int
        lr: float
        device: str
        pretrained_model: str
    Returns:
        None
    """
    team_id = "DQNAgent"
    
    def __init__(self, agent_id: int, input_spec, num_actions: int, lr: float=1e-3, device: str="cpu", pretrained_model=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.device = device
        self.gamma = 0.99
        self.lr = lr
        self.global_step = 0
        self.epsilon = 1.0

        # Networks: Q-Network (learning) and Target-Network (stable target)
        if pretrained_model:
            self.load_agent(pretrained_model)
        else:
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.q_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)

        self.target_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Sync weights initially
        
        self.loss_fn = nn.MSELoss()

    def act(self, map_state, aux_state, epsilon=0.0):
        """
        Take an action based on the state.
        Args:
            map_state: np.ndarray
            aux_state: np.ndarray
            epsilon: float
        Returns:
            action: int
        """
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.q_net(map_tensor, aux_tensor).argmax().item()
            
        # action with the highest predicted Q-value
        return action

    def act_masked(self, map_state, aux_state, action_mask, epsilon=0.0):
        allowed = self._mask_to_actions(action_mask)
        if random.random() < epsilon:
            return random.choice(allowed)

        map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(map_tensor, aux_tensor)[0]
            mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, -1e9)
            return int(q_values.argmax().item())

    def _mask_to_actions(self, action_mask):
        allowed = [int(i) for i, value in enumerate(np.asarray(action_mask)) if value > 0]
        return allowed or [0]

    def train_step(self, map_state, aux_state, next_map_state, next_aux_state, action, reward, done, next_action_mask=None):
        """
        Train the DQN agent for one step.
        Args:
            state: np.ndarray
            action: int
            reward: float
            next_state: np.ndarray
            done: bool
        Returns:
            None
        """
        # torch.from_numpy is zero-copy; only move to device when not CPU
        map_state_t      = torch.from_numpy(map_state)
        aux_state_t      = torch.from_numpy(aux_state)
        next_map_state_t = torch.from_numpy(next_map_state)
        next_aux_state_t = torch.from_numpy(next_aux_state)
        action_t     = torch.from_numpy(action).unsqueeze(1)
        reward_t     = torch.from_numpy(reward).unsqueeze(1)
        done_t       = torch.from_numpy(done).unsqueeze(1)
        next_action_mask_t = None
        if next_action_mask is not None:
            next_action_mask_t = torch.from_numpy(next_action_mask.astype(np.bool_))
        if self.device != "cpu":
            map_state_t      = map_state_t.to(self.device)
            aux_state_t      = aux_state_t.to(self.device)
            next_map_state_t = next_map_state_t.to(self.device)
            next_aux_state_t = next_aux_state_t.to(self.device)
            action_t     = action_t.to(self.device)
            reward_t     = reward_t.to(self.device)
            done_t       = done_t.to(self.device)
            if next_action_mask_t is not None:
                next_action_mask_t = next_action_mask_t.to(self.device)

        # 2. Calculate current Q-values: Q(s, a)
        # gather() extracts the Q-value for the specific action taken
        q_values = self.q_net(map_state_t, aux_state_t).gather(1, action_t)

        # max(1)[0] gets the max Q-value for the next state
            # ~ max_a' {Q(s', a', weights)}
        # If done=1, the future reward is 0.
            # Q*(s, a) = E[r + gamma * max_a' {Q*(s', a')}]
            # ~ Q(s, a) = r + gamma * max_a' {Q(s', a', weights)} if not done else Q(s, a) = r
        # inference_mode is stricter than no_grad: disables autograd engine entirely
        with torch.no_grad():
            next_q = self.target_net(next_map_state_t, next_aux_state_t)
            if next_action_mask_t is not None:
                next_q = next_q.masked_fill(~next_action_mask_t, -1e9)
            max_next_q = next_q.max(1)[0].unsqueeze(1)
            target_q   = reward_t + self.gamma * max_next_q * (1 - done_t)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad(set_to_none=True)  # skip memset, just nullify refs
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return loss.item()
        
    def update_target_network(self):
        """Copies the learned weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def load_agent(self, pretrained_model):
        checkpoint = torch.load(pretrained_model, map_location=self.device)
        input_spec = checkpoint.get("input_spec", checkpoint.get("input_shape", checkpoint["input_dim"]))
        self.map_shape = tuple(input_spec[0])
        self.aux_dim = int(input_spec[1])
        self.num_actions = checkpoint["num_actions"]
        self.q_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions).to(self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epsilon = checkpoint["epsilon"]

def train_dqn(user_id=0, enemy_type="simple", num_episodes=100, max_steps=500, seed=86, save_model=True, pretrained_model=None):
    # Training-only imports - placed here so they don't run when the evaluator loads this file
    import importlib.util as _importlib_util
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parent.parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from reward import compute_reward  
    from utils import (plot_loss, plot_rewards, plot_win_rates, 
                       plot_moving_average, seed_everything, save_model_fn)
    from agent import (SimpleRuleAgent, SmarterRuleAgent, 
                       TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent)
    from engine import BomberEnv 

    def _load_external_agent(agent_path, agent_id):
        agent_path = _Path(agent_path).resolve()
        if not agent_path.exists():
            raise FileNotFoundError(f"External enemy agent not found: {agent_path}")
        agent_dir = str(agent_path.parent)
        if agent_dir not in _sys.path:
            _sys.path.insert(0, agent_dir)
        spec = _importlib_util.spec_from_file_location("my_rule_enemy_agent", str(agent_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load external enemy agent: {agent_path}")
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "Agent"):
            raise AttributeError(f"External enemy agent must define class Agent: {agent_path}")
        return module.Agent(agent_id)

    env = BomberEnv(max_steps=max_steps, seed=seed)
    if enemy_type == "simple":
        enemy_agent = SimpleRuleAgent(1)
    elif enemy_type == "smarter":
        enemy_agent = SmarterRuleAgent(1)
    elif enemy_type == "tactical":
        enemy_agent = TacticalRuleAgent(1)
    elif enemy_type == "genius":
        enemy_agent = GeniusRuleAgent(1)
    elif enemy_type == "box_farmer":
        enemy_agent = BoxFarmerAgent(1)
    elif enemy_type == "my_rule":
        submission_agent_path = _root / "submission" / "agent.py"
        enemy_agent = _load_external_agent(submission_agent_path, 1)
    else:
        raise ValueError(f"Invalid enemy type: {enemy_type}")

    # hyperparam
    epsilon_start      = 1.0
    epsilon_min        = 0.05
    epsilon_decay      = 0.995
    epsilon            = epsilon_start
    batch_size         = 64
    lr                 = 1e-3

    dummy_obs = env.reset(seed=seed)
    agent_ids = [user_id, enemy_agent.agent_id]
    sample_state = encode_obs(dummy_obs, agent_ids=agent_ids)
    input_spec = (sample_state[0].shape, sample_state[1].shape[0])
    num_actions = 6

    user_agent = TrainingAgent(user_id, input_spec, num_actions, lr=lr, device="cuda" if torch.cuda.is_available() else "cpu", pretrained_model=pretrained_model)
    safety_agent = Agent(user_id, load_model=False)
    buffer = ReplayBuffer(capacity=10_000, map_shape=input_spec[0], aux_dim=input_spec[1], num_actions=num_actions)

    global_step = 0
    loss_history = []
    reward_history = []
    win_history = []
    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            prev_obs = None
            total_reward = 0

            map_state, aux_state = encode_obs(obs, agent_ids)

            for _ in range(max_steps):
                # 1. Action
                action_mask = safety_agent.action_mask(obs)
                user_action  = user_agent.act_masked(map_state, aux_state, action_mask, epsilon=epsilon)
                enemy_action = enemy_agent.act(obs)
                actions = [None, None]
                actions[user_id]              = user_action
                actions[enemy_agent.agent_id] = enemy_action

                # 2. Environment Step
                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated

                # 3. Reward
                r = compute_reward(prev_obs, next_obs, agent_id=user_id)
                total_reward += r
                reward_history.append(r)
                if done:
                    win_history.append(1 if next_obs["players"][user_id][2] else 0)
                
                # 4. Buffer Push
                next_map_state, next_aux_state = encode_obs(next_obs, agent_ids)
                next_action_mask = safety_agent.action_mask(next_obs)
                buffer.push(
                    map_state,
                    aux_state,
                    user_action,
                    r,
                    next_map_state,
                    next_aux_state,
                    done,
                    action_mask=action_mask,
                    next_action_mask=next_action_mask,
                )

                # 5. Train
                global_step += 1
                if len(buffer) >= batch_size:
                    (
                        sampled_map_state,
                        sampled_aux_state,
                        sampled_next_map_state,
                        sampled_next_aux_state,
                        sampled_action_mask,
                        sampled_next_action_mask,
                        sampled_action,
                        sampled_reward,
                        sampled_done,
                    ) = buffer.sample(batch_size)
                    loss = user_agent.train_step(
                        sampled_map_state,
                        sampled_aux_state,
                        sampled_next_map_state,
                        sampled_next_aux_state,
                        sampled_action,
                        sampled_reward,
                        sampled_done,
                        next_action_mask=sampled_next_action_mask,
                    )
                    loss_history.append(loss)

                # 6. Update
                prev_obs  = obs
                obs       = next_obs
                map_state = next_map_state
                aux_state = next_aux_state

                # 7. Done
                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if ep % 10 == 0:
                user_agent.update_target_network()
            pbar.update(1)
            pbar.set_postfix(reward=f"{total_reward:.2f}", epsilon=f"{epsilon:.3f}")

    model_folder = f"ckpts/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    if save_model:
        model_path = f"{model_folder}/{user_agent.global_step}_global_step.pth"
        save_model_fn(user_agent.q_net, 
                    user_agent.optimizer, 
                    user_agent.global_step, 
                    user_agent.epsilon, 
                    user_agent.lr, 
                    input_spec,
                    num_actions,
                    model_path)
        
    plot_loss(loss_history=loss_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_loss.png")
    plot_rewards(reward_history=reward_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_rewards.png")
    plot_win_rates(win_history=win_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_win_rates.png")
    plot_moving_average(data=reward_history, window_size=10, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_moving_average.png")

def training():
    from utils import seed_everything
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--enemy_type", type=str, default="simple", choices=["simple", "smarter", "tactical", "genius", "box_farmer", "my_rule"])
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=86, help="Random seed for reproducibility")
    parser.add_argument("--save_model", action="store_true", help="Save model")
    parser.add_argument("--load_model", type=str, default=None, help="Load model")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    print("Skip training? ", args.skip_training)
    if not args.skip_training:
        train_dqn(enemy_type=args.enemy_type, 
                    num_episodes=args.num_episodes, 
                    max_steps=args.max_steps, 
                    seed=args.seed, 
                    save_model=args.save_model,
                    pretrained_model=args.load_model)
    
# Mandatory for submission
class Agent:
    """Fast full-state agent used by the submitted `agent.py` interface."""

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    WALKABLE = {Map.GRASS, Map.ITEM_RADIUS, Map.ITEM_CAPACITY}
    BOMB_TIMER = 7

    def __init__(self, agent_id: int, load_model: bool = True):
        self.agent_id = int(agent_id)
        self.device = torch.device("cpu")
        self.q_net = None
        self.map_shape = (9, 13, 13)
        self.aux_dim = 3
        self.num_actions = 6
        self.model_ready = False

        checkpoint_path = Path(__file__).parent / "model.pth"
        if load_model:
            try:
                self._load_checkpoint(checkpoint_path)
                self.model_ready = True
            except Exception:
                # The rule layer below is a complete legal fallback if the checkpoint
                # is absent or incompatible in a local/submission environment.
                self.model_ready = False

    def act(self, obs: dict) -> int:
        try:
            return self._hybrid_action(obs)
        except Exception:
            return 0

    def _hybrid_action(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) == 0:
            return 0

        row, col, _, bombs_left, radius_bonus = players[self.agent_id]
        my_pos = (int(row), int(col))
        radius = min(Player.MAX_BOMB_RADIUS, 1 + int(radius_bonus))
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and int(p[2]) == 1
        ]
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        danger = self._danger_schedule(grid, players, bombs)
        valid_actions = self._valid_actions(grid, my_pos, bomb_positions)

        # Hard safety layer: never let the model override urgent bomb evasion.
        if self._is_threatened(my_pos, danger):
            escape = self._escape_action(grid, my_pos, bombs, danger)
            if escape is not None:
                return escape

        dqn_action = self._dqn_masked_action(obs, grid, players, bombs, my_pos, enemies, radius, danger)
        if dqn_action is not None:
            return dqn_action

        return self._rule_action(obs)

    def _rule_action(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) == 0:
            return 0

        row, col, _, bombs_left, radius_bonus = players[self.agent_id]
        my_pos = (int(row), int(col))
        radius = min(Player.MAX_BOMB_RADIUS, 1 + int(radius_bonus))
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and int(p[2]) == 1
        ]
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        danger = self._danger_schedule(grid, players, bombs)
        valid_actions = self._valid_actions(grid, my_pos, bomb_positions)

        if int(bombs_left) > 0 and my_pos not in bomb_positions:
            enemy_value = self._enemy_blast_value(grid, my_pos, enemies, radius)
            box_value = self._box_blast_value(grid, my_pos, radius)
            early_farm = int(radius_bonus) < 2 or int(bombs_left) < 2
            if (enemy_value > 0 or box_value >= (1 if early_farm else 2)) and self._can_escape_new_bomb(
                grid, players, bombs, my_pos, radius
            ):
                return 5

        items = self._item_targets(grid, bombs_left=int(bombs_left), radius_bonus=int(radius_bonus))
        move = self._move_to_targets(grid, my_pos, items, bombs, danger)
        if move is not None:
            return move

        attack_spots = self._attack_spots(grid, enemies, radius, bomb_positions)
        move = self._move_to_targets(grid, my_pos, attack_spots, bombs, danger)
        if move is not None:
            return move

        box_spots = self._box_bomb_spots(grid, bomb_positions)
        move = self._move_to_targets(grid, my_pos, box_spots, bombs, danger)
        if move is not None:
            return move

        move = self._pressure_action(grid, my_pos, enemies, bombs, danger)
        if move is not None:
            return move

        safe_actions = [
            action
            for action in valid_actions
            if not self._danger_at(self._next_pos(my_pos, action), danger, 1)
        ]
        if safe_actions:
            return max(safe_actions, key=lambda action: self._mobility(grid, self._next_pos(my_pos, action), bomb_positions))
        return 0

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        input_spec = checkpoint.get("input_spec", checkpoint.get("input_shape", checkpoint["input_dim"]))
        self.map_shape = tuple(input_spec[0])
        self.aux_dim = int(input_spec[1])
        self.num_actions = int(checkpoint["num_actions"])
        self.q_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.q_net.to(self.device)
        self.q_net.eval()

    def _dqn_masked_action(self, obs, grid, players, bombs, my_pos, enemies, radius, danger):
        if not self.model_ready or self.q_net is None:
            return None

        allowed = self._safe_actions_for_model(grid, players, bombs, my_pos, enemies, radius, danger)
        if not allowed:
            return None

        try:
            map_state, aux_state = encode_obs(obs, [self.agent_id])
            map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
            aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.q_net(map_tensor, aux_tensor)[0].detach().cpu().numpy()

            ranked_actions = sorted(allowed, key=lambda action: float(q_values[action]), reverse=True)
            chosen = ranked_actions[0]
            if chosen == 0 and len(ranked_actions) > 1 and not self._is_threatened(my_pos, danger):
                return ranked_actions[1]
            return int(chosen)
        except Exception:
            return None

    def _safe_actions_for_model(self, grid, players, bombs, my_pos, enemies, radius, danger):
        row, col, _, bombs_left, _ = players[self.agent_id]
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        allowed = []

        for action in self._valid_actions(grid, my_pos, bomb_positions):
            npos = self._next_pos(my_pos, action)
            if not self._danger_at(npos, danger, 1):
                allowed.append(action)

        if int(bombs_left) > 0 and my_pos not in bomb_positions:
            enemy_value = self._enemy_blast_value(grid, my_pos, enemies, radius)
            box_value = self._box_blast_value(grid, my_pos, radius)
            if (enemy_value > 0 or box_value > 0) and self._can_escape_new_bomb(
                grid, players, bombs, my_pos, radius
            ):
                allowed.append(5)

        return sorted(set(action for action in allowed if 0 <= action <= 5))

    def action_mask(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        mask = np.zeros(6, dtype=np.float32)

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) == 0:
            mask[0] = 1.0
            return mask

        row, col, _, _, radius_bonus = players[self.agent_id]
        my_pos = (int(row), int(col))
        radius = min(Player.MAX_BOMB_RADIUS, 1 + int(radius_bonus))
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and int(p[2]) == 1
        ]
        danger = self._danger_schedule(grid, players, bombs)

        if self._is_threatened(my_pos, danger):
            escape = self._escape_action(grid, my_pos, bombs, danger)
            if escape is not None:
                mask[int(escape)] = 1.0
                return mask

        allowed = self._safe_actions_for_model(grid, players, bombs, my_pos, enemies, radius, danger)
        for action in allowed:
            mask[int(action)] = 1.0
        if not np.any(mask):
            mask[0] = 1.0
        return mask

    def _next_pos(self, pos, action):
        drow, dcol = self.MOVES[action]
        return pos[0] + drow, pos[1] + dcol

    def _in_bounds(self, grid, row, col):
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

    def _passable(self, grid, row, col):
        return self._in_bounds(grid, row, col) and int(grid[row, col]) in self.WALKABLE

    def _bomb_rows(self, bombs):
        return [
            (int(b[0]), int(b[1]), max(1, int(b[2])), int(b[3]) if len(b) > 3 else -1)
            for b in bombs
        ]

    def _valid_actions(self, grid, pos, bomb_positions):
        actions = [0]
        for action in (1, 2, 3, 4):
            row, col = self._next_pos(pos, action)
            if self._passable(grid, row, col) and (row, col) not in bomb_positions:
                actions.append(action)
        return actions

    def _blast_tiles(self, grid, row, col, radius):
        tiles = {(row, col)}
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for distance in range(1, radius + 1):
                nrow, ncol = row + drow * distance, col + dcol * distance
                if not self._in_bounds(grid, nrow, ncol) or int(grid[nrow, ncol]) == Map.WALL:
                    break
                tiles.add((nrow, ncol))
                if int(grid[nrow, ncol]) == Map.BOX:
                    break
        return tiles

    def _bomb_radius(self, players, owner_id):
        if 0 <= owner_id < len(players):
            return min(Player.MAX_BOMB_RADIUS, 1 + int(players[owner_id][4]))
        return 2

    def _danger_schedule(self, grid, players, bombs, extra_bomb=None):
        bomb_rows = [
            (row, col, timer, owner_id, self._bomb_radius(players, owner_id))
            for row, col, timer, owner_id in self._bomb_rows(bombs)
        ]
        if extra_bomb is not None:
            bomb_rows.append(extra_bomb)

        blast_sets = [
            self._blast_tiles(grid, row, col, radius)
            for row, col, _, _, radius in bomb_rows
        ]
        timers = [timer for _, _, timer, _, _ in bomb_rows]

        # Chain reaction timing reaches a fixed point in a few bombs on a 13x13 board.
        changed = True
        while changed:
            changed = False
            for i, (row, col, _, _, _) in enumerate(bomb_rows):
                for j, blast in enumerate(blast_sets):
                    if i != j and (row, col) in blast and timers[j] < timers[i]:
                        timers[i] = timers[j]
                        changed = True

        danger = {}
        for timer, blast in zip(timers, blast_sets):
            for tile in blast:
                danger.setdefault(tile, set()).add(timer)
        return danger

    def _danger_at(self, pos, danger, step):
        return step in danger.get(pos, ())

    def _is_threatened(self, pos, danger):
        timers = danger.get(pos, ())
        return any(timer <= self.BOMB_TIMER for timer in timers)

    def _bomb_blocked_at(self, pos, bombs, step):
        for row, col, timer, _ in self._bomb_rows(bombs):
            if pos == (row, col) and step <= timer:
                return True
        return False

    def _escape_action(self, grid, start, bombs, danger):
        queue = deque([(start, 0, None)])
        seen = {(start, 0)}
        best = None

        while queue:
            pos, step, first_action = queue.popleft()
            if step > 0 and not self._is_threatened(pos, danger):
                return first_action
            if step >= self.BOMB_TIMER:
                continue

            for action in (1, 2, 3, 4, 0):
                npos = self._next_pos(pos, action)
                nstep = step + 1
                if action != 0 and not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos != pos and self._bomb_blocked_at(npos, bombs, nstep):
                    continue
                if self._danger_at(npos, danger, nstep):
                    continue
                state = (npos, nstep)
                if state in seen:
                    continue
                seen.add(state)
                candidate_first = action if first_action is None else first_action
                if best is None and candidate_first != 0:
                    best = candidate_first
                queue.append((npos, nstep, candidate_first))
        return best

    def _can_escape_new_bomb(self, grid, players, bombs, pos, radius):
        simulated_bombs = list(self._bomb_rows(bombs))
        simulated_bombs.append((pos[0], pos[1], self.BOMB_TIMER, self.agent_id))
        danger = self._danger_schedule(
            grid,
            players,
            bombs,
            extra_bomb=(pos[0], pos[1], self.BOMB_TIMER, self.agent_id, radius),
        )
        return self._survives_until_bomb_blast(grid, pos, simulated_bombs, danger)

    def _survives_until_bomb_blast(self, grid, start, bombs, danger):
        queue = deque([(start, 0)])
        seen = {(start, 0)}
        while queue:
            pos, step = queue.popleft()
            if step >= self.BOMB_TIMER and not self._danger_at(pos, danger, step):
                return True
            if step >= self.BOMB_TIMER:
                continue
            for action in (1, 2, 3, 4, 0):
                npos = self._next_pos(pos, action)
                nstep = step + 1
                if action != 0 and not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos != pos and self._bomb_blocked_at(npos, bombs, nstep):
                    continue
                if self._danger_at(npos, danger, nstep):
                    continue
                state = (npos, nstep)
                if state not in seen:
                    seen.add(state)
                    queue.append(state)
        return False

    def _move_to_targets(self, grid, start, targets, bombs, danger):
        if not targets:
            return None
        bomb_positions = {(row, col) for row, col, _, _ in self._bomb_rows(bombs)}
        queue = deque([(start, None, 0)])
        seen = {start}
        while queue:
            pos, first_action, distance = queue.popleft()
            if pos in targets and first_action is not None:
                return first_action
            for action in (1, 2, 3, 4):
                npos = self._next_pos(pos, action)
                if npos in seen or not self._passable(grid, npos[0], npos[1]):
                    continue
                if npos in bomb_positions or self._danger_at(npos, danger, min(distance + 1, self.BOMB_TIMER)):
                    continue
                seen.add(npos)
                queue.append((npos, action if first_action is None else first_action, distance + 1))
        return None

    def _item_targets(self, grid, bombs_left, radius_bonus):
        want_capacity = bombs_left <= 1
        want_radius = radius_bonus <= 1
        preferred = set()
        fallback = set()
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                cell = int(grid[row, col])
                if cell not in (Map.ITEM_RADIUS, Map.ITEM_CAPACITY):
                    continue
                fallback.add((row, col))
                if (cell == Map.ITEM_CAPACITY and want_capacity) or (cell == Map.ITEM_RADIUS and want_radius):
                    preferred.add((row, col))
        return preferred or fallback

    def _box_blast_value(self, grid, pos, radius):
        return sum(int(grid[row, col]) == Map.BOX for row, col in self._blast_tiles(grid, pos[0], pos[1], radius))

    def _enemy_blast_value(self, grid, pos, enemies, radius):
        blast = self._blast_tiles(grid, pos[0], pos[1], radius)
        return sum(enemy in blast for enemy in enemies)

    def _box_bomb_spots(self, grid, bomb_positions):
        spots = set()
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if int(grid[row, col]) != Map.BOX:
                    continue
                for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nrow, ncol = row + drow, col + dcol
                    if self._passable(grid, nrow, ncol) and (nrow, ncol) not in bomb_positions:
                        spots.add((nrow, ncol))
        return spots

    def _attack_spots(self, grid, enemies, radius, bomb_positions):
        spots = set()
        for erow, ecol in enemies:
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                for distance in range(1, radius + 1):
                    row, col = erow + drow * distance, ecol + dcol * distance
                    if not self._in_bounds(grid, row, col) or int(grid[row, col]) == Map.WALL:
                        break
                    if int(grid[row, col]) == Map.BOX:
                        break
                    if self._passable(grid, row, col) and (row, col) not in bomb_positions:
                        spots.add((row, col))
        return spots

    def _pressure_action(self, grid, start, enemies, bombs, danger):
        if not enemies:
            return None
        bomb_positions = {(row, col) for row, col, _, _ in self._bomb_rows(bombs)}
        choices = []
        for action in self._valid_actions(grid, start, bomb_positions):
            npos = self._next_pos(start, action)
            if self._danger_at(npos, danger, 1):
                continue
            distance = min(abs(npos[0] - enemy[0]) + abs(npos[1] - enemy[1]) for enemy in enemies)
            choices.append((distance, -self._mobility(grid, npos, bomb_positions), action))
        return min(choices)[2] if choices else None

    def _mobility(self, grid, pos, bomb_positions):
        return len(self._valid_actions(grid, pos, bomb_positions))
        

if __name__ == "__main__":
    training()
