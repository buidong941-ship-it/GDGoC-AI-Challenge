# GDGoC AI Challenge 2026 - Bomberland: Competition Guide
---

## Table of Contents

1. [Game Overview](#1-game-overview)
2. [Game Mechanics](#2-game-mechanics)
3. [Registration & Submission](#3-registration--submission)
4. [Agent Structure & Submission Requirements](#4-agent-structure--submission-requirements)
5. [Scoring System & Leaderboard](#5-scoring-system--leaderboard)
6. [Baseline Agents](#6-baseline-agents)
7. [RL Resources](#7-rl-resources)
8. [Prizes & Timeline](#8-prizes--timeline)
9. [Grand Finals](#9-grand-finals)
10. [Evaluation Server Configuration](#10-evaluation-server-configuration)
11. [Participant Kit](#11-participant-kit)
12. [Transparency](#12-transparency)

---

## 1. Game Overview

**Bomberland** is a tactical strategy game inspired by the classic [Bomb IT](https://gamevui.vn/bom-it-7/game). 4 agents compete on a **13×13** grid map (playing area of 11×11). Agents' mission: move, place bombs, destroy boxes, collect items, and eliminate opponents.

### Map

- Size: **13×13** (11x11 play area, 1 tile thick border of walls).
- Randomly generated per match using a `seed` - guaranteed to be **fully connected** (no isolated areas).
- Contains 4 types of tiles:

| Symbol | Name | Description |
|---|---|---|
| `0` | Grass | Walkable, bombs can be placed here |
| `1` | Wall | Indestructible, blocks explosions |
| `2` | Box | Can be destroyed by bombs; may drop items when destroyed |
| `3` | Item: Radius | Collect to increase bomb radius by +1 |
| `4` | Item: Capacity | Collect to increase bomb capacity by +1 |

### Starting Positions

The 4 agents start at the **4 corners** of the map:

- Agent 0: `(1, 1)` - Top-left
- Agent 1: `(11, 11)` - Bottom-right
- Agent 2: `(1, 11)` - Top-right
- Agent 3: `(11, 1)` - Bottom-left

The 2×2 area around each corner is guaranteed to be empty (no boxes, no walls) to prevent agents from being stuck at the start.

---

## 2. Game Mechanics

### Resolution

Each game step is processed in the following order:

```
Collect Actions → Process Movement → Place Bombs → Decrease Bomb Timers → Resolve Explosions → Remove Agents → Spawn Items → Check End Conditions
```

### Agent Actions

Each step, the agent returns **1 integer**:

| Value | Action |
|---|---|
| `0` | STOP (stay still) |
| `1` | LEFT |
| `2` | RIGHT |
| `3` | UP |
| `4` | DOWN |
| `5` | PLACE_BOMB |

**Movement Rules:**
- Cannot move through walls (`1`) or boxes (`2`).
- **Cannot move through bombs** from previous steps (bombs already on the floor) except for when an agent standing on the same tile where they just placed a bomb can still move away.
- Multiple agents **can stand on the same tile**.
- If multiple agents step on an item tile simultaneously → the item is destroyed (no one gets it).

### Bomb Mechanics

**Default Parameters:**
- Timer: **7 steps** (counts down from 7 to 0, explodes when ≤ 0)
- Default Radius: **1** (increases with collected items)
- Initial Capacity: **1** (increases with items)
- Maximum Radius: **5**, Maximum Capacity: **5**

**Placement Rules:**
- Can only place if `bombs_left > 0`.
- Cannot place on a tile that already has a bomb from a previous step.
- If multiple agents place a bomb on the **same tile in the same step** → the engine processes agents by ID (0 → 3). A bomb from a later agent **only overwrites** the previous one if its radius is **strictly larger**. Specific results:
  - Different radii → keep the bomb with the **larger radius** (regardless of agent ID).
  - Equal radii → keep the bomb of the **agent with the smaller ID** (as they were processed first).
  - Only the **owner of the kept bomb** will have their `bombs_left` decremented.
- Only the **placer** consumes `bombs_left`. The bomb remains and will explode even if the owner is eliminated.

**Explosion Mechanics:**
- Bombs explode in **4 directions** (Up, Down, Left, Right) with the set radius.
- Explosions **stop at walls** (cannot pass through).
- Explosions **stop at boxes** and **destroy them** (cannot pass through further).
- Explosions **pass through agents** (not blocked by players).
- Explosions last for only **1 step**.
- **Chain reaction**: If an explosion touches another bomb, that bomb explodes immediately in the same step.

**When a Box is Destroyed:**
- 30% chance to drop `Item Radius` (+1 bomb radius)
- 30% chance to drop `Item Capacity` (+1 bomb capacity)
- 40% chance to drop nothing

**Auto-spawning Items:**
Each step, every empty grass tile (no agent present) has a small chance to spawn an item:
- Probability increases over time: `P = 0.0003 × (step / 165)`
- 50% Radius, 50% Capacity

**When an Agent is Eliminated:**
- Any agent standing on a tile affected by an explosion is **immediately eliminated**.
- Eliminated agents can no longer act. However, bombs placed before death remain on the map and explode as usual.

**Self-destruct:** Agent can self destroyed :- ).

### Match Conclusion

The match ends when:
- **≤ 1 agent remains alive** (`terminated = True`), or
- Reaches **500 steps** (`truncated = True`)

---

## 3. Registration & Submission

### Step 1: Team Registration

Complete the [Registration Form](https://forms.gle/2m3GUehGNnZR5pnLA) with:
- Team Name
- Primary contact name & email
- Secondary contact name & email (optional)
- Agreement to competition rules

Upon successful registration, the system will send a **confirmation email** to the primary contact, containing:

| Information | Description |
|---|---|
| `Team Name` | Your registered team name |
| `Canonical Team ID` | Unique identifier (used for submissions) |
| `Submission Token` | Authentication password for submissions (do not share) |
| Submission Form Link | Link to submit your agent |
| Discord/Contact Link | Support channel |

> **Registering again with the same (team name, primary contact email)** will generate a new `Submission Token` and invalidate the old one.

### Step 2: Submission

Complete the [Submission Form](https://forms.gle/2peRHZCbEjy3WjQj9) with:
- `Canonical Team ID` (from email)
- `Submission Token` (from email)
- Changelog (description of changes)
- `.zip` file uploaded to Google Drive

**Submission Limit:** **3 times per day** (resets at 7:00 AM UTC+7).

---

## 4. Agent Structure & Submission Requirements

### ZIP File Structure

The submission must be a `.zip` file containing **exactly one `agent.py`** file:

```
submission.zip
├── agent.py          ← Required, unique
├── model.pth         ← (optional, for DRL)
└── ...               ← (optional, weights, small configs, etc.)
```

### File Limits

| Criteria | Limit |
|---|---|
| ZIP file size | ≤ 100 MB |
| Total extracted size | ≤ 300 MB |
| Single file size | ≤ 150 MB |
| Max number of files | ≤ 20 files |
| Allowed extensions | `.py`, `.txt`, `.pt`, `.pth`, `.pkl`, `.onnx`, `.bin`, `.json`, `.yaml`, `.yml`, `.md`, `.h5`, `.pb`, `.keras`, `.tflite` |

### Required Agent Class

The `agent.py` file **must** define an `Agent` class with the following interface:

```python
class Agent:
    def __init__(self, agent_id: int):
        # agent_id: 0, 1, 2, or 3
        self.agent_id = agent_id

    def act(self, obs: dict) -> int:
        # obs: dict containing 'map', 'players', 'bombs'
        # Returns: int in [0, 5]
        ...
```

**`agent_id`?**
- An integer from `0` to `3` assigned automatically by the engine. It tells your agent which player it is controlling. 
- Use `self.agent_id` to index into the `obs["players"]` list to find your own coordinates, health, and bomb stats. For example, `my_data = obs["players"][self.agent_id]`.

### Observation (obs) received each step

```python
obs = {
    "map":     np.ndarray,  # shape (13, 13), dtype int
                            # 0=Grass, 1=Wall, 2=Box, 3=Item_Radius, 4=Item_Capacity
    "players": np.ndarray,  # shape (4, 5), dtype int8
                            # Each row: [row, col, alive, bombs_left, bomb_radius_bonus]
    "bombs":   np.ndarray,  # shape (N, 4), dtype int8, N = current number of bombs
                            # Each row: [row, col, timer, owner_id]
}
```

**Notes:**
- `alive`: 1 = alive, 0 = eliminated
- `bomb_radius_bonus`: bonus added to radius (Actual radius = 1 + bonus)
- `bombs_left`: number of bombs available to place
- Agents receive **full state**.

### Time Constraints

- **Startup timeout:** Agent must finish loading within **20 seconds**.
- **Inference timeout:** Each call to `act()` must return within **100ms**. If exceeded → action defaults to `0` (STOP).

### Important Rules

- ❌ **No LLMs allowed** (GPT, Gemini, Claude, etc.) inside the `Agent` class.
- ❌ **No copying code** from other teams.
- ❌ **Import Restrictions:** You may only `import` libraries that are pre-installed in the evaluation environment (see `requirements.txt`). Any attempt to import an unlisted library (e.g., `import langchain`) will cause your agent to crash.
- ✅ **Major Libraries Allowed:** `numpy`, `scipy`, `torch` (PyTorch), `tensorflow`, `stable-baselines3`, `gymnasium`, `onnxruntime`, and all Python standard libraries.
- ✅ **Checkpoints:** Allowed to load pre-trained models from files inside your ZIP (e.g., `.pth`, `.onnx`, `.h5`, `.tflite`).
- ✅ **Independent Submissions:** Each submission is treated as an agent with its own rating.

---

## 5. Scoring System & Leaderboard

### Scoring: TrueSkill

The system uses the [**TrueSkill**](https://trueskill.org/) algorithm (Microsoft Research) to estimate the skill level of each agent. Each submission is characterized by:

- **μ (mu):** Estimated average skill
- **σ (sigma):** Uncertainty (sigma decreases as more matches are played)
- **Score = μ - 3σ:** Conservative score used for ranking - ensures agents with enough matches are ranked higher.

Default starting values: `μ = 100.0, σ = 33.333`.

### Match Results

Each match has 4 agents. Ranking in the match is determined by **elimination order**:
- Last agent alive (or surviving until the end) = Best rank
- First agent eliminated = Worst rank
- Agents eliminated in the same step → Draw (shared rank)

**Step 500 Tie-breaker:**
If multiple agents survive to step 500, they are no longer guaranteed a draw. Surviving agents are ranked based on their active engagement during the match, in the following hierarchy:
1. Most Kills
2. Most Boxes Destroyed
3. Most Items Collected
4. Most Bombs Placed

If agents have exactly identical statistics across all 4 metrics, they will share a draw.

**Win/Draw/Loss:**
Determined by match ranking:
- **Win:** Achieving the unique best rank in the match (e.g., last one alive, or sole survivor at time limit).
- **Draw:** Sharing the best rank with other agents (e.g., multiple agents eliminated in the same final step, or multiple survivors at 500 steps).
- **Loss:** Failing to achieve the best rank (eliminated before the winner).

### Leaderboard

Ranking priority (tie-break):

1. **Higher** `Score = μ − 3σ`
2. If equal: **Higher** `μ`
3. If still equal: **Lower** `σ`
4. If still equal: **Most recent submission**

### Automated Evaluation System

**Immediately upon submission:**
1. Server downloads ZIP from Google Drive.
2. Structure and syntax check.
3. If valid → immediately trigger **a batch of 12 matches** for initial rating. Opponents are sampled from the Active Pool (Ratio: 40% similar rating, 30% top tier, 30% random).
4. Update Google Sheets leaderboard.

**Background Job (Continuous):**
- Workers run continuously in cycles: each cycle runs **5 matches**, followed by a 10-second rest.
- Each match is sampled from the **Active Pool**, ensuring at least 1 student agent is involved.

### Active Pool

The system automatically maintains a pool of eligible agents for matchmaking:

| Criteria | Description |
|---|---|
| All Baselines | Always in the pool |
| Best per Team | Each team's highest-scoring submission |
| Recent per Team | Each team's 2 most recent submissions |
| Top Global | Top 10 global submissions (with ≥ 10 matches) |

---

## 6. Baseline Agents

There are **6 rule-based baseline agents** with fixed ratings (do not change during the competition):

| Name | Strategy | Score (μ − 3σ) |
|---|---|---|
| `tactical_rule_agent` | Dodges danger, finds items, targets enemies, calculates bomb placement | ~114.7 |
| `genius_rule_agent` | Balanced offense/defense, BFS pathfinding | ~112.5 |
| `smarter_rule_agent` | Prioritizes boxes, dodges bombs, chases enemies | ~111.3 |
| `box_farmer_agent` | Focuses on breaking boxes for items | ~107.9 |
| `simple_rule_agent` | Simple rules: dodge bombs, place bombs | ~107.8 |
| `random_agent` | Random actions | ~99.0 |

Additionally, an **RL-based agent** (`dqn_agent`) is provided as a reference.

---

## 7. RL Resources

1. [Reinforcement Learning: An Introduction, Andrew Barto and Richard S. Sutton](http://incompleteideas.net/book/the-book-2nd.html)
2. [Reinforcement Learning Course, University of Toronto](https://bereyhi-courses.github.io/rl-utoronto/)
3. [Neuriton](https://www.facebook.com/share/1B4BpDUbyx/)
4. [Reinforcement Learning Exercises, AI VIETNAM](https://www.facebook.com/share/p/192LyNsvbp/)
5. [RL Algorithms Single File Codes](https://github.com/vwxyzjn/cleanrl)

---

## 8. Prizes & Timeline

### Prizes (Estimated)

| Rank | Prize |
|---|---|
| 🥇 1st Place | 500,000 VNĐ |
| 🥈 2nd Place | 400,000 VNĐ |
| 🥉 3rd Place | 300,000 VNĐ |
| 🏅 4th Place | 200,000 VNĐ |
| 🎖️ 5th Place | 100,000 VNĐ |


### Timeline

| Date | Event |
|---|---|
| 20/5 | Registration & Submissions open |
| 24/5 | Workshop about the competition & RL |
| 21/6 | **Submissions close** |
| 21-22/6 | Top 8 selected → Grand Finals → Results Announcement |
| 24/6 | Pitching & Award ceremony |

**Expected Scale:** 25-30 teams, open to external participants.

---

## 9. Grand Finals

### Selecting Top K

After freezing the leaderboard, **Top 8 teams** are selected based on their best submission, prioritized by:

1. Highest Score (`μ − 3σ`)
2. Higher `μ`
3. Lower `σ`

One **best baseline agent** will also participate as a benchmark.

### Grand Finals Format: Round Robin

- All 4-player combinations from the finalist pool are listed (`C(9,4)` combinations).
- Each combination runs **50 matches**.
- Points awarded based on match rank:
  - 1st Rank (Best): **3 points**
  - 2nd Rank: **2 points**
  - 3rd Rank: **1 point**
  - 4th Rank (Worst): **0 points**

### Grand Finals Tie-break

If total points are tied:
1. Frozen Leaderboard Score (`μ − 3σ`)
2. Higher `μ`
3. Lower `σ`

---

## 10. Evaluation Server Configuration

| Parameter | Detail |
|---|---|
| Server | Google Cloud VM (`e2-standard-8`) |
| CPU | 8 vCPUs (4 physical cores) |
| RAM | 32 GB |
| OS | Ubuntu 22.04 LTS |
| Python | 3.11 (Conda environment) |
| Inference timeout | 100ms/step |
| Startup timeout | 20 seconds |
| Parallel workers | Up to 6 simultaneous matches |
| Max steps/match | 500 steps |

Agents run in **isolated processes** - they cannot affect other agents or the evaluation system. If an agent crashes or times out → action defaults to `STOP (0)`.

---

## 11. Participant Kit

The Participant Kit is a **public** repository containing everything needed to develop your agent.

**Repo Link:** [Link](https://github.com/VLTisME/Bomberland-GDGoC-AI-Challenge).

---

## 12. Transparency

### Fixed Seeds

Each match is assigned a **unique random seed**. This seed is saved in the JSON log - allowing anyone to reproduce the match using the same engine and agents.

### Public Logs

Every match is saved as:
- **JSON File** (`logs/json/match_*.json`): Complete step history, seed, and final ranking.
- **GIF File** (`logs/gifs/match_*.gif`): Match animation (if enabled).

These files are uploaded to [Google Drive](https://drive.google.com/drive/folders/1FBNTDpOJh_eMOIoe18in_e1myBQuUm0i?usp=sharing) and linked in the Google Sheets leaderboard for public viewing.

### Leaderboard

The leaderboard is updated on [Google Sheets](https://docs.google.com/spreadsheets/d/1caRS0zqKovKqsL5ozzqNAtSWhseTVBT1LNBr0AVDrBE/edit?usp=sharing).

### Open Source

The entire game engine and baseline agents are public in the Participant Kit, ensuring everyone can verify the game logic.

---

*This document will be updated as changes occur. Contact us via Discord or email for any questions.*
