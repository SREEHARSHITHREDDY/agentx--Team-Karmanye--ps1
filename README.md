# agentx--Team-Karmanye--ps1

# AgentX — Adaptive Maze Reinforcement Learning (PS1)

## Team
**Team Name:** Team Karmanye  
**Problem Statement:** PS1 – Adaptive Agent in a Dynamic Environment

---

## Problem Statement
The objective of Problem Statement 1 (PS1) is to design an adaptive intelligent agent
that can learn optimal behavior in a dynamic and uncertain environment.

AgentX addresses this by applying Reinforcement Learning to a grid-based maze
environment, where the agent learns optimal navigation policies through interaction,
reward feedback, and iterative improvement.

---

## Approach Overview
AgentX is implemented using a model-free reinforcement learning approach.

- **Environment:** Grid-based maze with free cells, obstacles, and a goal state
- **Agent:** Q-Learning agent
- **State Space:** Agent position represented as `(row, column)`
- **Action Space:** Up, Down, Left, Right
- **Learning Strategy:** ε-greedy exploration and exploitation
- **Adaptation:** Retraining and transfer learning across different maze sizes
- **Evaluation:** Greedy policy execution after training

---

## Algorithms Used
- Q-Learning
- ε-Greedy Exploration Strategy
- Greedy Policy Evaluation
- Transfer Learning using historical Q-tables
- BFS-based Maze Solvability Checking

---

## Features Implemented
- Random solvable M×N maze generation
- Built-in example mazes for quick demonstration
- Automated training reward tracking
- Greedy path visualization
- Training convergence estimation
- Automatic report generation (JSON & TXT)
- Cross-run performance comparison
- Persistent run history tracking

---

## Setup Instructions
```bash
git clone https://github.com/SREEHARSHITHREDDY/agentx--Team-Karmanye--ps1.git
cd agentx--Team-Karmanye--ps1
pip install -r requirements.txt
