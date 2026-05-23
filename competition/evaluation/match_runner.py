import os
import json
from datetime import datetime

from competition.integrations.drive_upload import upload_file_to_drive
from competition.evaluation.rendering import render_match_frame
from engine.game import BomberEnv
from competition.evaluation.runtime_guard import AgentProcessExecutor


class MatchRunner:
    def __init__(self, log_dir='logs', enable_gif=True):
        self.log_dir = log_dir
        self.enable_gif = enable_gif
        os.makedirs(log_dir, exist_ok=True)
        if self.enable_gif:
            os.makedirs(os.path.join(log_dir, 'gifs'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'json'), exist_ok=True)

    def run_match(
        self,
        agent_paths,
        team_ids,
        seed=None,
        max_steps=500,
        inference_timeout_s=0.1,
        startup_timeout_s: float | None = None,
    ):
        if len(agent_paths) != len(team_ids):
            raise ValueError(
                f"agent_paths and team_ids length mismatch: {len(agent_paths)} vs {len(team_ids)}"
            )

        if startup_timeout_s is None:
            startup_timeout_env = os.getenv("EVALUATION_STARTUP_TIMEOUT_S")
            startup_timeout_s = float(startup_timeout_env) if startup_timeout_env else max(1.0, inference_timeout_s * 10)

        env = BomberEnv(seed=seed, max_steps=max_steps)
        executors = []
        failed_loads = []
        for i, path in enumerate(agent_paths):
            executor = AgentProcessExecutor(agent_path=path, agent_id=i)
            try:
                executor.start(startup_timeout_s=startup_timeout_s)
                executors.append(executor)
            except Exception as e:
                failed_loads.append((i, path, str(e)))

        if failed_loads:
            details = ", ".join([f"idx={i}, path={path}, err={err}" for i, path, err in failed_loads])
            for executor in executors:
                executor.terminate()
            raise RuntimeError(f"Failed to load agents: {details}")
        
        obs = env.reset(seed=seed)
        frames = []
        history = []
        
        terminated = False
        truncated = False
        
        # Initial frame and history snapshot
        initial_obs = {
            "map": obs["map"].tolist(),
            "players": obs["players"].tolist(),
            "bombs": obs["bombs"].tolist(),
            "_step": env.current_step,
        }
        history.append({
            "step": env.current_step,
            "actions": None,
            "alive": [bool(p.alive) for p in env.players],
            "map": initial_obs["map"],
            "players": initial_obs["players"],
            "bombs": initial_obs["bombs"],
        })
        if self.enable_gif:
            frames.append(render_match_frame(initial_obs, prev_obs=None, agent_metadata={"agent_names": team_ids}))
        
        # Track death order for ranking
        death_order = []
        ranks = [0] * len(executors)
        alive_mask = [True] * len(executors)
        survival_steps = [0] * len(executors)
        runtime_stats = {
            str(i): {
                "timeouts": 0,
                "errors": 0,
                "invalid_actions": 0,
                "fallback_uses": 0,
            }
            for i in range(len(executors))
        }

        try:
            while not (terminated or truncated):
                prev_obs = obs
                actions = []
                for i, executor in enumerate(executors):
                    if env.players[i].alive:
                        result = executor.act_with_timeout(obs=obs, timeout_s=inference_timeout_s)
                        action = result.action

                        if result.timeout:
                            runtime_stats[str(i)]["timeouts"] += 1
                        if result.error and not result.timeout:
                            runtime_stats[str(i)]["errors"] += 1
                        if result.invalid_action:
                            runtime_stats[str(i)]["invalid_actions"] += 1
                        if action == 0 and (result.timeout or result.error or result.invalid_action):
                            runtime_stats[str(i)]["fallback_uses"] += 1

                        actions.append(action)
                    else:
                        actions.append(0)

                obs, terminated, truncated = env.step(actions)
            
                # Record state for JSON
                history.append({
                    "step": env.current_step,
                    "actions": actions,
                    "alive": [bool(p.alive) for p in env.players],
                    "map": obs["map"].tolist(),
                    "players": obs["players"].tolist(),
                    "bombs": obs["bombs"].tolist(),
                })
            
                if self.enable_gif:
                    # Render frame for GIF when enabled.
                    frame_obs = {
                        "map": obs["map"].tolist(),
                        "players": obs["players"].tolist(),
                        "bombs": obs["bombs"].tolist(),
                        "_step": env.current_step,
                    }
                    prev_frame_obs = {
                        "map": prev_obs["map"].tolist(),
                        "players": prev_obs["players"].tolist(),
                        "bombs": prev_obs["bombs"].tolist(),
                        "_step": env.current_step - 1,
                    }
                    frames.append(render_match_frame(frame_obs, prev_obs=prev_frame_obs, agent_metadata={"agent_names": team_ids}))
            
                deaths = []
                for i, p in enumerate(env.players):
                    if alive_mask[i] and not p.alive:
                        alive_mask[i] = False
                        survival_steps[i] = env.current_step
                        deaths.append(i)
                # death_order = [[1, 2], [3]] meaning 1 and 2 died at the same time, then 3, 0 is still alive. Or [[1]] then only 1 died, 0, 2, 3 are still alive and they can all survive until the end and gets rank 0.
                if deaths:
                    death_order.append(deaths)
        finally:
            for executor in executors:
                executor.terminate()

        # determine who's alive = rank 0
        alives = []
        for i, p in enumerate(env.players):
            if alive_mask[i] and p.alive:
                alives.append(i)
                alive_mask[i] = False
                survival_steps[i] = env.current_step
        if alives:
            def get_stats(i):
                return (
                    env.players[i].stats['kills'],
                    env.players[i].stats['boxes'],
                    env.players[i].stats['items'],
                    env.players[i].stats['bombs']
                )
            alives.sort(key=get_stats, reverse=True)
            
            groups = []
            current_group = [alives[0]]
            current_stats = get_stats(alives[0])
            for i in alives[1:]:
                stats = get_stats(i)
                if stats == current_stats:
                    current_group.append(i)
                else:
                    groups.append(current_group)
                    current_group = [i]
                    current_stats = stats
            groups.append(current_group)
            
            # groups has best first. death_order expects worst first, so extend reversed.
            death_order.extend(reversed(groups))
        
        # Determine final ranks, backward
        for rank, group in enumerate(reversed(death_order)):
            for i in group:
                ranks[i] = rank

        # Save logs
        match_name = f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"
        
        gif_path = None
        if self.enable_gif and frames:
            gif_path = os.path.join(self.log_dir, 'gifs', f"{match_name}.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
        
        # Save JSON
        json_path = os.path.join(self.log_dir, 'json', f"{match_name}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "seed": seed,
                "team_ids": team_ids,
                "meta": {"agent_names": team_ids},
                "ranks": ranks,
                "survival_steps": survival_steps,
                "runtime_stats": runtime_stats,
                "history": history
            }, f)

        json_drive_url = None
        gif_drive_url = None
        drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
        if drive_folder_id:
            try:
                json_upload = upload_file_to_drive(None, drive_folder_id, json_path)
                json_drive_url = json_upload.get("web_view_link")
            except Exception:
                json_drive_url = None

            if gif_path:
                try:
                    gif_upload = upload_file_to_drive(None, drive_folder_id, gif_path)
                    gif_drive_url = gif_upload.get("web_view_link")
                except Exception:
                    gif_drive_url = None
            
        return ranks, survival_steps, gif_path, json_path, gif_drive_url, json_drive_url
    
    
if __name__ == "__main__":
    # testing
    runner = MatchRunner()
    agent_paths = [
        # "submissions/team_alpha/20260326_120000/agent.py",
        # "submissions/team_beta/20260326_120000/agent.py",
        "agent/smarter_rule_agent.py",
        "agent/genius_rule_agent.py",
        "agent/tactical_rule_agent.py",
        "agent/random_agent.py"
    ]
    team_ids = ["SmarterRuleAgent", "GeniusRuleAgent", "TacticalRuleAgent", "RandomAgent"]
    ranks, survival_steps, gif_path, json_path, gif_drive_url, json_drive_url = runner.run_match(agent_paths, team_ids, seed=45)