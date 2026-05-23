import os
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch

parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent
from training import encode_obs, DQNAgent, DQfDAgent
from training.SQIL import encode_obs as sqil_encode_obs
from training.bc_ppo_lstm import BC_PPO_LSTM_Agent, is_bc_ppo_lstm_checkpoint
from training.bc_ppo_lstm_attn_selfplay import ActorCriticAttnLSTM
from competition.evaluation.runtime_guard import load_agent_instance

class Viewer:
	PLAYER_COLORS = [(220, 50, 50), (50, 50, 220), (30, 150, 30), (200, 140, 0)]

	def __init__(self, width=13, height=13, cell_size=42, fps=8, panel_width=200):
		self.width = width
		self.height = height
		self.cell_size = cell_size
		self.fps = fps
		self.panel_width = panel_width

		self.top_bar = 60
		self.grid_width = width * cell_size
		self.screen_width = self.grid_width + panel_width
		self.screen_height = height * cell_size + self.top_bar

		pygame.init()
		self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
		pygame.display.set_caption("Bomberland Enhanced Viewer")
		self.clock = pygame.time.Clock()
		self.font_info = pygame.font.SysFont(None, 24)
		self.font_small = pygame.font.SysFont(None, 20)
		self.explosion_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
		self.explosion_overlay.fill((255, 140, 0, 130))

	def draw_grid(self, grid):
		for row in range(self.height):
			for col in range(self.width):
				rect = pygame.Rect(
					col * self.cell_size,
					row * self.cell_size + self.top_bar,
					self.cell_size,
					self.cell_size,
				)
				cell_type = int(grid[row, col])
				if cell_type == 1:
					pygame.draw.rect(self.screen, (80, 80, 80), rect)
					pygame.draw.rect(self.screen, (40, 40, 40), rect, 2)
				elif cell_type == 2:
					pygame.draw.rect(self.screen, (139, 69, 19), rect)
					pygame.draw.rect(self.screen, (101, 67, 33), rect, 2)
					pygame.draw.line(self.screen, (101, 67, 33), (rect.left, rect.top), (rect.right, rect.bottom), 2)
					pygame.draw.line(self.screen, (101, 67, 33), (rect.right, rect.top), (rect.left, rect.bottom), 2)
				elif cell_type == 3:
					pygame.draw.rect(self.screen, (225, 225, 225), rect)
					pygame.draw.circle(self.screen, (255, 0, 0), rect.center, self.cell_size // 4)
					text = self.font_small.render("R", True, (255, 255, 255))
					self.screen.blit(text, (rect.centerx - 5, rect.centery - 8))
				elif cell_type == 4:
					pygame.draw.rect(self.screen, (225, 225, 225), rect)
					pygame.draw.circle(self.screen, (0, 0, 255), rect.center, self.cell_size // 4)
					text = self.font_small.render("C", True, (255, 255, 255))
					self.screen.blit(text, (rect.centerx - 5, rect.centery - 8))
				else:
					pygame.draw.rect(self.screen, (144, 238, 144), rect)
					pygame.draw.rect(self.screen, (120, 200, 120), rect, 1)

	def draw_players(self, players):
		for i, p in enumerate(players):
			if p[2] != 1:
				continue
			center = (
				int(p[1]) * self.cell_size + self.cell_size // 2,
				int(p[0]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, self.PLAYER_COLORS[i % len(self.PLAYER_COLORS)], center, self.cell_size // 3)
			img = self.font_small.render(str(i), True, (255, 255, 255))
			self.screen.blit(img, (center[0] - 5, center[1] - 8))
			stats_text = f"B:{int(p[3])} R:{int(p[4])}"
			stats_img = self.font_small.render(stats_text, True, (0, 0, 0))
			self.screen.blit(stats_img, (center[0] - 16, center[1] + 12))

	def draw_bombs(self, bombs):
		for b in bombs:
			if b[2] <= 0:
				continue
			center = (
				int(b[1]) * self.cell_size + self.cell_size // 2,
				int(b[0]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, (20, 20, 20), center, self.cell_size // 4)
			timer_img = self.font_small.render(str(int(b[2])), True, (255, 255, 255))
			self.screen.blit(timer_img, (center[0] - 5, center[1] - 8))

	def draw_agent_sidebar(self, players, agent_names):
		"""Right panel: agent name, alive/dead, bombs available, radius power-up bonus."""
		x0 = self.grid_width
		pygame.draw.rect(self.screen, (52, 58, 64), (x0, 0, self.panel_width, self.screen_height))
		pygame.draw.line(self.screen, (30, 30, 30), (x0, 0), (x0, self.screen_height), 2)

		title = self.font_info.render("Agents", True, (245, 245, 245))
		self.screen.blit(title, (x0 + 10, self.top_bar + 8))

		y = self.top_bar + 40
		line_h = 22
		for i, p in enumerate(players):
			name = agent_names[i] if i < len(agent_names) and agent_names[i] else f"Agent {i}"
			alive = int(p[2]) == 1
			bombs_left = int(p[3])
			radius_bonus = int(p[4])
			color = self.PLAYER_COLORS[i % len(self.PLAYER_COLORS)]

			pygame.draw.circle(self.screen, color, (x0 + 14, y + 8), 6)
			name_img = self.font_small.render(str(name)[:28], True, (240, 240, 240))
			self.screen.blit(name_img, (x0 + 28, y))
			y += line_h

			status = "Alive" if alive else "Dead"
			status_color = (120, 220, 140) if alive else (220, 100, 100)
			status_img = self.font_small.render(status, True, status_color)
			self.screen.blit(status_img, (x0 + 10, y))
			y += line_h

			stats = f"Bombs: {bombs_left}  |  +Radius: {radius_bonus}"
			stats_img = self.font_small.render(stats, True, (200, 200, 200))
			self.screen.blit(stats_img, (x0 + 10, y))
			y += line_h + 10

	def draw_header(self, episode_idx, total_episodes, step_idx, total_steps, paused):
		pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.screen_width, self.top_bar))
		status = "PAUSED" if paused else "PLAYING"
		text = (
			f"Ep {episode_idx + 1}/{total_episodes} | "
			f"Step {step_idx}/{max(total_steps - 1, 0)} | {status}"
		)
		help_text = "[A/D] Step [W/S] Ep [SPACE] Pause [ESC] Quit"
		self.screen.blit(self.font_info.render(text, True, (245, 245, 245)), (10, 5))
		self.screen.blit(self.font_small.render(help_text, True, (210, 210, 210)), (10, 35))

	def _in_bounds(self, row, col):
		return 0 <= row < self.height and 0 <= col < self.width

	def _blast_tiles(self, grid, bx, by, radius):
		tiles = {(bx, by)}
		for drow, dcol in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
			for r in range(1, radius + 1):
				tr, tc = bx + drow * r, by + dcol * r
				if not self._in_bounds(tr, tc):
					break
				cell = int(grid[tr, tc])
				if cell == 1:
					break
				tiles.add((tr, tc))
				if cell == 2:
					break
		return tiles

	def _explosion_tiles_from_transition(self, prev_obs, obs):
		if prev_obs is None:
			return set()

		prev_bombs = prev_obs["bombs"]
		curr_bombs = obs["bombs"]
		curr_positions = {(int(b[0]), int(b[1])) for b in curr_bombs}
		prev_players = prev_obs["players"]
		prev_grid = prev_obs["map"]

		tiles = set()
		for b in prev_bombs:
			bx, by, timer, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
			exploded = timer <= 1 or (bx, by) not in curr_positions
			if not exploded:
				continue
			radius = 1
			if 0 <= owner_id < len(prev_players):
				radius = 1 + int(prev_players[owner_id][4])
			tiles.update(self._blast_tiles(prev_grid, bx, by, radius))
		return tiles

	def draw_explosions(self, explosion_tiles):
		for row, col in explosion_tiles:
			px = col * self.cell_size
			py = row * self.cell_size + self.top_bar
			self.screen.blit(self.explosion_overlay, (px, py))
			center = (px + self.cell_size // 2, py + self.cell_size // 2)
			pygame.draw.circle(self.screen, (255, 220, 120), center, self.cell_size // 6)

	def render(self, obs, prev_obs, episode_idx, total_episodes, step_idx, total_steps, paused, agent_names):
		self.screen.fill((245, 245, 245))
		self.draw_grid(obs["map"])
		explosion_tiles = self._explosion_tiles_from_transition(prev_obs, obs)
		self.draw_explosions(explosion_tiles)
		self.draw_players(obs["players"])
		self.draw_bombs(obs["bombs"])
		self.draw_agent_sidebar(obs["players"], agent_names)
		self.draw_header(episode_idx, total_episodes, step_idx, total_steps, paused)
		pygame.display.flip()
		self.clock.tick(self.fps)

	def close(self):
		pygame.quit()


def str2bool(value):
	if isinstance(value, bool):
		return value
	value = str(value).strip().lower()
	if value in {"true", "1", "yes", "y", "t"}:
		return True
	if value in {"false", "0", "no", "n", "f"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def is_bc_ppo_attn_selfplay_checkpoint(ckpt: dict) -> bool:
	"""True for checkpoints produced by training/bc_ppo_lstm_attn_selfplay.py."""
	if ckpt.get("agent_type") == "bc_ppo_lstm_attn_selfplay":
		return True
	meta = ckpt.get("meta")
	if isinstance(meta, dict) and meta.get("model_variant") in {"lstm", "attn", "attn_lstm"} and (
		meta.get("input_spec") is not None or ckpt.get("input_shape") is not None
	):
		return True
	return False


class BC_PPO_AttnSelfplay_Agent:
	"""Greedy / ε-greedy wrapper (supports lstm/attn/attn_lstm variants)."""

	def __init__(
		self,
		agent_id: int,
		checkpoint_path: str,
		device: str | None = None,
		force_variant: str | None = None,
	):
		self.agent_id = int(agent_id)
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		ckpt = torch.load(checkpoint_path, map_location=self.device)
		meta = ckpt.get("meta", {})
		input_spec = meta.get("input_spec") or ckpt.get("input_shape")
		if input_spec is None:
			raise ValueError(f"Checkpoint {checkpoint_path!r} missing meta['input_spec'] or input_shape")
		map_shape = tuple(input_spec[0])
		aux_dim = int(input_spec[1])
		num_actions = int(meta.get("num_actions", ckpt.get("num_actions", 6)))
		variant = force_variant or meta.get("model_variant", "lstm")
		if variant not in {"lstm", "attn", "attn_lstm"}:
			raise ValueError(f"Invalid model variant {variant!r} for {checkpoint_path!r}")

		self.map_shape = map_shape
		self.aux_dim = aux_dim
		self.num_actions = num_actions
		self.variant = variant

		self.model = ActorCriticAttnLSTM(
			map_shape=map_shape,
			aux_dim=aux_dim,
			num_actions=num_actions,
			variant=variant,
			map_feat_dim=int(meta.get("map_feat_dim", 128)),
			aux_embed_dim=int(meta.get("aux_embed_dim", 32)),
			lstm_hidden=int(meta.get("lstm_hidden", 128)),
			lstm_layers=int(meta.get("lstm_layers", 1)),
			attn_d_model=int(meta.get("attn_d_model", 128)),
			attn_heads=int(meta.get("attn_heads", 4)),
			pos_max_h=int(meta.get("pos_max_h", 32)),
			pos_max_w=int(meta.get("pos_max_w", 32)),
		).to(self.device)
		self.model.load_state_dict(ckpt["model_state_dict"])
		self.model.eval()
		self._hidden = None

	def reset_memory(self) -> None:
		self._hidden = None

	def act(self, map_state, aux_state, epsilon: float = 0.0) -> int:
		m = torch.from_numpy(map_state).float().unsqueeze(0).to(self.device)
		aux = torch.from_numpy(aux_state).float().unsqueeze(0).to(self.device)
		with torch.no_grad():
			if self.model.use_lstm:
				if self._hidden is None:
					self._hidden = self.model.init_hidden(1, self.device)
				logits, _, self._hidden = self.model.forward_step(m, aux, self._hidden)
			else:
				logits, _, _ = self.model.forward_step(m, aux, None)
		if random.random() < float(epsilon):
			return random.randint(0, self.num_actions - 1)
		return int(logits.argmax(dim=-1).item())


def make_agents(agent_paths, seed=None):
	n_players = len(agent_paths)
	agents = [None] * n_players
	names = [None] * n_players

	if seed is not None:
		random.seed(seed)

	for i, path in enumerate(agent_paths):
		if path == "None" or path.lower() == "random":
			# Random rule-based baseline
			x = random.randint(0, 5)
			if x == 0:
				names[i] = "RandomAgent"
				agents[i] = RandomAgent(i)
			elif x == 1:
				names[i] = "SimpleRuleAgent"
				agents[i] = SimpleRuleAgent(i)
			elif x == 2:
				names[i] = "SmarterRuleAgent"
				agents[i] = SmarterRuleAgent(i)
			elif x == 3:
				names[i] = "GeniusRuleAgent"
				agents[i] = GeniusRuleAgent(i)
			elif x == 4:
				names[i] = "BoxFarmerAgent"
				agents[i] = BoxFarmerAgent(i)
			else:
				names[i] = "TacticalRuleAgent"
				agents[i] = TacticalRuleAgent(i)
		elif path == "RandomAgent":
			names[i] = "RandomAgent"
			agents[i] = RandomAgent(i)
		elif path == "SimpleRuleAgent":
			names[i] = "SimpleRuleAgent"
			agents[i] = SimpleRuleAgent(i)
		elif path == "SmarterRuleAgent":
			names[i] = "SmarterRuleAgent"
			agents[i] = SmarterRuleAgent(i)
		elif path == "GeniusRuleAgent":
			names[i] = "GeniusRuleAgent"
			agents[i] = GeniusRuleAgent(i)
		elif path == "BoxFarmerAgent":
			names[i] = "BoxFarmerAgent"
			agents[i] = BoxFarmerAgent(i)
		elif path == "TacticalRuleAgent":
			names[i] = "TacticalRuleAgent"
			agents[i] = TacticalRuleAgent(i)
		else:
			# Custom agent path
			p = Path(path)
			if p.is_dir():
				p = p / "agent.py"
			if not p.exists():
				raise FileNotFoundError(f"Agent file not found: {p}")
			
			try:
				agents[i] = load_agent_instance(str(p), i)
				if hasattr(agents[i], "team_id"):
					names[i] = agents[i].team_id
				else:
					names[i] = p.parent.name if p.parent.name else p.name
			except Exception as e:
				raise RuntimeError(f"Failed to load agent from {p}: {e}")

	return agents, names


def clone_obs(obs):
	return {
		"map": np.array(obs["map"], copy=True),
		"players": np.array(obs["players"], copy=True),
		"bombs": np.array(obs["bombs"], copy=True),
	}


ACTION_MIRROR = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 5}


def rotate_map_180(map_feat):
	"""Flip spatial features both vertically and horizontally (180-degree rotation)
	so a bottom-right agent sees the board as if it were at top-left."""
	return np.flip(map_feat, axis=(1, 2)).copy()


ACTION_FLIP_H = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 5}  # mirror columns: L<->R
ACTION_FLIP_V = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5}  # mirror rows: U<->D


def orient_map_to_topleft(map_feat, agent_id):
	"""
	Re-orient spatial features so the given agent's starting corner is viewed as top-left.

	Assumes the standard 4-corner spawn layout:
	- 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
	For other ids, leaves the map unchanged.
	"""
	if agent_id == 0:
		return map_feat
	if agent_id == 1:
		return np.flip(map_feat, axis=2).copy()
	if agent_id == 2:
		return np.flip(map_feat, axis=1).copy()
	if agent_id == 3:
		return rotate_map_180(map_feat)
	return map_feat


def unorient_action_from_topleft(action, agent_id):
	"""
	Map an action chosen in the "agent at top-left" orientation back to the env's frame.
	"""
	if agent_id == 0:
		return action
	if agent_id == 1:
		return ACTION_FLIP_H[action]
	if agent_id == 2:
		return ACTION_FLIP_V[action]
	if agent_id == 3:
		return ACTION_MIRROR[action]
	return action


def _expected_input_spec(agent):
	"""Return (map_channels, aux_dim) if available, else (None, None)."""
	map_shape = getattr(agent, "map_shape", None)
	aux_dim = getattr(agent, "aux_dim", None)
	if map_shape is None or aux_dim is None:
		return None, None
	return int(map_shape[0]), int(aux_dim)


def _encode_for_model_agent(obs, agent_id, all_agent_ids, agent):
	"""Encode obs using a feature format compatible with the loaded model."""
	map_channels, aux_dim = _expected_input_spec(agent)

	# SQIL checkpoints in this repo use 11 map channels + 5 aux scalars.
	if map_channels == 11 and aux_dim == 5:
		return sqil_encode_obs(obs, agent_ids=[agent_id, *all_agent_ids])

	# Fallback to legacy DQN encoding (9 channels + 3 aux).
	opp_id = all_agent_ids[0] if all_agent_ids else (1 - agent_id)
	return encode_obs(obs, agent_ids=[agent_id, opp_id])


def simulate_episodes(agent_paths, num_episodes=10, max_steps=500, seed=None, model_variants=None):
	env = BomberEnv(max_steps=max_steps)
	agents, names = make_agents(agent_paths, seed=seed)
	
	episodes = []
	num_agents = len(agents)

	for episode in range(num_episodes):
		episode_seed = None if seed is None else seed + episode
		obs = env.reset(seed=episode_seed)
		done = False
		step = 0
		trajectory = [clone_obs(obs)]

		while not done and step < max_steps:
			actions = []
			for i in range(num_agents):
				try:
					action = agents[i].act(obs)
				except Exception as e:
					print(f"Agent {names[i]} failed to act: {e}")
					action = 0
				actions.append(action)
				
			obs, terminated, truncated = env.step(actions)
			trajectory.append(clone_obs(obs))
			done = terminated or truncated
			step += 1

		episodes.append(trajectory)

	return episodes, names


def run_simple_viewer(agent_paths, num_episodes=10, max_steps=100, seed=None, autoplay=True, model_variants=None):
	episodes, agent_names = simulate_episodes(
		agent_paths=agent_paths,
		num_episodes=num_episodes,
		max_steps=max_steps,
		seed=seed,
		model_variants=model_variants,
	)
	if not episodes:
		print("No episodes to display.")
		return

	first_obs = episodes[0][0]
	viewer = Viewer(width=first_obs["map"].shape[1], height=first_obs["map"].shape[0])

	print("Agents:", ", ".join(agent_names))
	print("Controls: A/D step, W/S episode, SPACE pause/play, ESC quit")

	episode_idx = 0
	step_idx = 0
	paused = not autoplay
	last_tick = time.time()

	running = True
	while running:
		now = time.time()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_SPACE:
					paused = not paused
				elif event.key == pygame.K_d:
					step_idx = min(step_idx + 1, len(episodes[episode_idx]) - 1)
					paused = True
				elif event.key == pygame.K_a:
					step_idx = max(step_idx - 1, 0)
					paused = True
				elif event.key == pygame.K_s:
					episode_idx = min(episode_idx + 1, len(episodes) - 1)
					step_idx = 0
				elif event.key == pygame.K_w:
					episode_idx = max(episode_idx - 1, 0)
					step_idx = 0

		if not paused and (now - last_tick) >= (1 / max(viewer.fps, 1)):
			if step_idx < len(episodes[episode_idx]) - 1:
				step_idx += 1
			else:
				paused = True
			last_tick = now

		current_obs = episodes[episode_idx][step_idx]
		prev_obs = episodes[episode_idx][step_idx - 1] if step_idx > 0 else None
		viewer.render(
			obs=current_obs,
			prev_obs=prev_obs,
			episode_idx=episode_idx,
			total_episodes=len(episodes),
			step_idx=step_idx,
			total_steps=len(episodes[episode_idx]),
			paused=paused,
			agent_names=agent_names,
		)

	viewer.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Local viewer for agents."
	)
	parser.add_argument("--agent_paths", nargs="+", default=["None", "None", "None", "None"])
	parser.add_argument("--num_episodes", type=int, default=10)
	parser.add_argument("--max_steps", type=int, default=500)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--autoplay", type=str2bool, default=True)
	args = parser.parse_args()

	run_simple_viewer(
		agent_paths=args.agent_paths,
		num_episodes=args.num_episodes,
		max_steps=args.max_steps,
		seed=args.seed,
		autoplay=args.autoplay,
	)
