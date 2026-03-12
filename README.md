# Multi-Agent Hide and Seek (MAPPO + Curiosity)

This project implements a **competitive multi-agent reinforcement learning environment** where two seeker agents attempt to locate a hiding agent in a MuJoCo physics world. The system combines **Multi-Agent Proximal Policy Optimization (MAPPO)**, **centralized value estimation**, **intrinsic curiosity exploration**, and **curriculum learning** to train agents that operate under partial observability. The entire training pipeline is designed for **large-scale parallel rollout collection** and robust physics simulation.

---

# 1. Environment Construction

The environment is constructed programmatically by generating a **MuJoCo MJCF XML specification**.

The generated world includes:

- A **20×20 bounded arena**
- Static outer walls
- An **internal room** creating occlusion
- A **movable box** introduced during later curriculum stages
- Three agents:
  - Seeker 1
  - Seeker 2
  - Hider

Each agent body is constructed with:

- Two **translation slide joints** (`x`, `y`)
- One **yaw hinge joint**
- A forward-facing geometry representing orientation

Translation is controlled by **high-gear motors**, while yaw is actuated via **separate rotational motors**.

This design ensures:

- simple locomotion
- physically consistent motion
- easy discrete control mapping.

---

# 2. Observation Model

The environment is **partially observable**.

Agents cannot access global state and instead rely on **local perception**.

Each agent observation consists of:
