import os
import random
import argparse
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
from competition.evaluation.runtime_guard import load_agent_instance


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


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
                # If agent has a team_id class attribute, use it, otherwise use folder name
                if hasattr(agents[i], "team_id"):
                    names[i] = agents[i].team_id
                else:
                    names[i] = p.parent.name if p.parent.name else p.name
            except Exception as e:
                raise RuntimeError(f"Failed to load agent from {p}: {e}")

    return agents, names


def run_match(agent_paths, num_episodes=10, max_steps=500, seed=None):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    n_players = len(agent_paths)
    
    agents, names = make_agents(agent_paths, seed)
    info = [{"name": names[i], "wins": 0} for i in range(n_players)]

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        obs = env.reset(seed=episode_seed)
        done = False
        step = 0
        death_order = []
        prev_alive = [bool(p[2]) for p in obs["players"]]

        while not done and step < max_steps:
            actions = []
            for i in range(n_players):
                try:
                    action = agents[i].act(obs)
                except Exception as e:
                    print(f"Agent {names[i]} failed to act: {e}")
                    action = 0
                actions.append(action)
                
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

            alive_now = [bool(p[2]) for p in obs["players"]]
            for i in range(n_players):
                if prev_alive[i] and not alive_now[i]:
                    death_order.append(info[i]["name"])
            prev_alive = alive_now
        
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [i for i in range(n_players) if alive_final[i]]
        
        if len(survivors) == 1:
            winner = survivors[0]
            info[winner]["wins"] += 1
            print(f"Episode {episode + 1}: {info[winner]['name']} wins | Died: {death_order}")
        else:
            print(f"Episode {episode + 1}: Draw | Died: {death_order}")

    print("\n=== Summary ===")
    for i in range(n_players):
        print(f"{info[i]['name']}: {info[i]['wins']} wins")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_paths", nargs="+", default=["None", "None", "None", "None"],
                        help="Paths to agent.py files, agent folders, or baseline names (e.g. RandomAgent). Use 'None' for a random baseline.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--autoplay", type=str2bool, default=True)
    args = parser.parse_args()
    
    if args.visualize:
        from scripts.participant.visualizer import run_simple_viewer

        run_simple_viewer(
            agent_paths=args.agent_paths,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            autoplay=args.autoplay,
        )
    else:
        run_match(
            agent_paths=args.agent_paths,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed
        )