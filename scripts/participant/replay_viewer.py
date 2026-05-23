import argparse
import importlib
import json
import sys
import time
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from competition.evaluation.rendering import render_match_frame

pygame = importlib.import_module("pygame")


class ReplayViewer:
    def __init__(self, history, meta=None, title="Bomberland Replay", fps=8):
        self.history = history
        self.meta = meta or {}
        self.fps = fps
        self.paused = False
        self.step_idx = 0
        self.last_tick = time.time()

        first = history[0]
        frame = render_match_frame(self._make_obs(first), prev_obs=None, agent_metadata=self.meta)
        self.screen_width, self.screen_height = frame.size

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.title = title
        self.font_small = pygame.font.SysFont(None, 22)

    def _make_obs(self, entry):
        return {
            "map": entry["map"],
            "players": entry["players"],
            "bombs": entry["bombs"],
            "_step": entry["step"],
        }

    def _render_surface(self, step_idx):
        current_entry = self.history[step_idx]
        prev_entry = self.history[step_idx - 1] if step_idx > 0 else None
        current_obs = self._make_obs(current_entry)
        prev_obs = self._make_obs(prev_entry) if prev_entry is not None else None
        image = render_match_frame(current_obs, prev_obs=prev_obs, agent_metadata=self.meta)
        return pygame.image.fromstring(image.tobytes(), image.size, image.mode)

    def run(self):
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
                        self.paused = not self.paused
                    elif event.key == pygame.K_d:
                        self.step_idx = min(self.step_idx + 1, len(self.history) - 1)
                        self.paused = True
                    elif event.key == pygame.K_a:
                        self.step_idx = max(self.step_idx - 1, 0)
                        self.paused = True
                    elif event.key == pygame.K_e:
                        self.step_idx = len(self.history) - 1
                        self.paused = True
                    elif event.key == pygame.K_q:
                        self.step_idx = 0
                        self.paused = True

            if not self.paused and (now - self.last_tick) >= (1 / max(self.fps, 1)):
                if self.step_idx < len(self.history) - 1:
                    self.step_idx += 1
                else:
                    self.paused = True
                self.last_tick = now

            surface = self._render_surface(self.step_idx)
            self.screen.blit(surface, (0, 0))

            help_text = "SPACE: Play/Pause | A: Prev Step | D: Next Step | Q: First | E: Last | ESC: Quit"
            help_img = self.font_small.render(help_text, True, (245, 245, 245))
            help_bg_h = help_img.get_height() + 8
            help_bg_y = self.screen_height - help_bg_h
            help_bg = pygame.Surface((self.screen_width, help_bg_h), pygame.SRCALPHA)
            help_bg.fill((0, 0, 0, 140))
            self.screen.blit(help_bg, (0, help_bg_y))
            self.screen.blit(help_img, (8, help_bg_y + 4))

            status = "PAUSED" if self.paused else "PLAYING"
            step = self.history[self.step_idx]["step"]
            pygame.display.set_caption(
                f"{self.title} | {status} | Step {step}/{self.history[-1]['step']} | SPACE pause/play | A/D step | Q/E jump"
            )
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def load_history(json_path):
    with open(json_path, "r") as handle:
        payload = json.load(handle)
    history = payload.get("history", [])
    meta = payload.get("meta") or {}
    if "agent_names" not in meta:
        team_ids = payload.get("team_ids") or []
        if team_ids:
            meta = {**meta, "agent_names": team_ids}
    return history, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to a match JSON file produced by evaluation/match_runner.py")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--paused", action="store_true", help="Start paused")
    args = parser.parse_args()

    history, meta = load_history(args.json_path)
    viewer = ReplayViewer(
        history=history,
        meta=meta,
        title=f"Bomberland Replay - {Path(args.json_path).name}",
        fps=args.fps,
    )
    viewer.paused = args.paused
    viewer.run()
