import os
import argparse
import random
import sys
from pathlib import Path

import trueskill

parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents


def estimate_rankings(agent_path, num_matches=100, max_steps=500):
    print(f"Loading agent from {agent_path}...")
    
    # Initialize TrueSkill environment with new competition defaults
    ts_env = trueskill.TrueSkill(mu=100.0, sigma=33.333, draw_probability=0.1)
    agent_rating = ts_env.Rating()
    baseline_rating = ts_env.Rating()

    wins = 0
    draws = 0
    total_rank = 0

    env = BomberEnv(max_steps=max_steps)

    for i in range(num_matches):
        # Player 0 is the agent, Player 1-3 are random baselines
        agent_paths = [agent_path, "Random", "Random", "Random"]
        try:
            agents, names = make_agents(agent_paths, seed=None)
        except Exception as e:
            print(f"Failed to load agent: {e}")
            return
            
        agent_name = names[0]
        
        obs = env.reset()
        done = False
        step = 0
        
        prev_alive = [bool(p[2]) for p in obs["players"]]
        death_order = []
        
        while not done and step < max_steps:
            actions = []
            for j in range(4):
                try:
                    action = agents[j].act(obs)
                except Exception:
                    action = 0
                actions.append(action)
                
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            
            alive_now = [bool(p[2]) for p in obs["players"]]
            for j in range(4):
                if prev_alive[j] and not alive_now[j]:
                    death_order.append(j)
            prev_alive = alive_now
            
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [j for j in range(4) if alive_final[j]]
        
        # Calculate ranks
        # Rank 0 is best. Death order: first to die gets worst rank.
        # survivors get rank 0.
        ranks = [0] * 4
        # Suppose death_order = [2, 1, 3] -> 2 died first, 1 died second, 3 died third. 0 survived.
        # 0 gets rank 0. 3 gets rank 1. 1 gets rank 2. 2 gets rank 3.
        for j in survivors:
            ranks[j] = 0
            
        current_rank = 1 if len(survivors) > 0 else 0
        for group in reversed(death_order):
            # death order is list of player indices who died one by one
            ranks[group] = current_rank
            current_rank += 1
            
        if len(survivors) == 1 and survivors[0] == 0:
            wins += 1
        elif len(survivors) > 1 and 0 in survivors:
            draws += 1
            
        total_rank += ranks[0]
        
        # Update TrueSkill
        rating_groups = [(agent_rating,), (baseline_rating,), (baseline_rating,), (baseline_rating,)]
        new_ratings = ts_env.rate(rating_groups, ranks=ranks)
        agent_rating = new_ratings[0][0]
        
        # Print progress
        score = agent_rating.mu - 3 * agent_rating.sigma
        print(f"Match {i+1}/{num_matches} | Rank: {ranks[0]} | Est. Score: {score:.2f} | mu: {agent_rating.mu:.2f}, sigma: {agent_rating.sigma:.2f}", end="\r")

    print("\n\n=== Final Estimated Results ===")
    print(f"Agent: {agent_name}")
    print(f"Matches Played: {num_matches}")
    print(f"Win Rate: {(wins / num_matches) * 100:.1f}%")
    print(f"Draw Rate: {(draws / num_matches) * 100:.1f}%")
    print(f"Average Rank: {total_rank / num_matches:.2f} (0 is winner, 3 is first to die)")
    print(f"Estimated TrueSkill: Score = {agent_rating.mu - 3 * agent_rating.sigma:.2f} (mu={agent_rating.mu:.2f}, sigma={agent_rating.sigma:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate agent rankings by playing against random baselines.")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to your agent.py file or agent folder.")
    parser.add_argument("--num_matches", type=int, default=100, help="Number of matches to simulate.")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per match.")
    args = parser.parse_args()
    
    estimate_rankings(args.agent_path, args.num_matches, args.max_steps)
