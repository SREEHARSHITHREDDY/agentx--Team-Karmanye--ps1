# agentx--Team-Karmanye--ps1

# AgentX — Adaptive Maze Reinforcement Learning (PS1)

## Team
**Team Name:** Team Karmanye  
**Problem Statement:** PS1 – Adaptive Agent in a Dynamic Environment

---

## Problem Statement
The objective of Problem Statement 1 (PS1) is to design an adaptive intelligent agent
that can learn optimal behavior in a dynamic and uncertain environment.

AgentX addresses this by using Reinforcement Learning to navigate grid-based mazes,
learning from interaction with the environment and adapting its policy over time.

---

## Approach Overview
AgentX is built using a model-free reinforcement learning approach.

- **Environment:** Grid-based maze with free cells, obstacles, and a goal
- **Agent:** Q-Learning based agent
- **State Space:** Agent position represented as (row, column)
- **Action Space:** Up, Down, Left, Right
- **Learning Strategy:** ε-greedy exploration with exploitation
- **Adaptation:** Retraining and transfer learning across different maze configurations

---

## Algorithms Used
- Q-Learning
- ε-Greedy Exploration Strategy
- Greedy Policy Evaluation
- Transfer Learning via Q-table reuse
- BFS-based Maze Solvability Check

---

## Setup Instructions
```bash
git clone https://github.com/<org>/agentx-team-karmanye-ps1.git
cd agentx-team-karmanye-ps1
pip install -r requirements.txt
