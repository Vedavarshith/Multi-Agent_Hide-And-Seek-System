# Multi-Agent_Hide-And-Seek-System

A multi-agent reinforcement learning system where two seeker agents attempt to find a hiding agent inside a MuJoCo physics environment.

The system implements a competitive multi-agent learning framework using:

MAPPO (Multi-Agent PPO)

Centralized Critic

Intrinsic Curiosity Module (ICM)

Curriculum Learning

Vectorized MuJoCo simulation

The objective is to observe emergent strategies as agents learn to search and hide within a partially observable environment.

System Overview

The environment contains three agents:

Agent	Role
Seeker 1	searches for hider
Seeker 2	searches for hider
Hider	avoids detection

Agents operate under partial observability using LIDAR-like raycasting sensors.

The learning system is structured as a competitive MARL game:

Seekers maximize:
    ability to detect and capture the hider

Hider maximizes:
    survival time and avoidance
Environment Design

The physics simulation is built using MuJoCo MJCF.

Key features:

Arena

A 20×20 bounded arena with walls.

+--------------------+
|                    |
|                    |
|      arena         |
|                    |
|                    |
+--------------------+
Internal Room

An inner room structure introduces occlusion and hiding locations.

      _______
     |       |
     | room  |
     |_______|
Dynamic Objects

A movable box appears in later curriculum levels.

Agents can use it to:

block line-of-sight

create hiding structures

Observation Space

Each agent receives a local observation vector:

obs = [
    velocity_x
    velocity_y
    sin(yaw)
    cos(yaw)
    lidar readings
]
LIDAR Encoding

For each ray:

(distance, is_hider, is_wall)

Total observation size:

obs_dim = 2 + 2 + (n_rays × 3)

Example:

n_rays = 21
obs_dim = 2 + 2 + 63 = 67

This gives agents limited perception similar to real robots.

Action Space

Each agent performs two discrete actions per step:

Movement
0 : no-op
1 : move +x
2 : move -x
3 : move +y
4 : move -y
Rotation
0 : no-op
1 : rotate left
2 : rotate right

Combined action representation:

(move, yaw)

Total environment action vector:

[s1_move, s1_yaw,
 s2_move, s2_yaw,
 h_move,  h_yaw]
Multi-Agent Learning Architecture

The algorithm follows MAPPO (Multi-Agent Proximal Policy Optimization).

Key design:

Decentralized Actors
Centralized Critic
Actors

Each agent has its own policy network.

Two actor types exist:

seeker_actor
hider_actor

Both share the same architecture.

Actor Network

The actor predicts two discrete policies:

π_move(a|s)
π_yaw(a|s)

Architecture:

Observation
     │
     ▼
Linear( obs_dim → 512 )
Tanh
     │
     ▼
Linear(512 → 512)
Tanh
     │
     ▼
Linear(512 → 256)
Tanh
     │
     ├── Move Head (5 logits)
     └── Yaw Head (3 logits)

Action sampling:

move ~ Categorical(move_logits)
yaw  ~ Categorical(yaw_logits)
Centralized Critic

While policies are decentralized, value estimation uses global state information.

Input:

critic_input =
    concat(
        global_state,
        one_hot(agent_id)
    )

Where

global_state = concat(obs_seeker1, obs_seeker2, obs_hider)

Architecture:

(global_state + agent_id)
        │
        ▼
Linear → 512
Tanh
        │
        ▼
Linear → 512
Tanh
        │
        ▼
Linear → 256
Tanh
        │
        ▼
Linear → 1

This allows the critic to estimate values using full system information.

MAPPO Optimization

Training follows the PPO clipped objective.

Policy ratio:

r(θ) = πθ(a|s) / πθ_old(a|s)

Loss:

L = min(
        r(θ) * A,
        clip(r(θ), 1-ε, 1+ε) * A
     )

Full objective:

Loss =
    policy_loss
  + value_coef * value_loss
  - entropy_coef * entropy

Advantages are computed using Generalized Advantage Estimation (GAE).

Advantage Estimation (GAE)

The algorithm computes advantages:

δ_t = r_t + γ V(s_{t+1}) - V(s_t)

A_t = δ_t + γλ A_{t+1}

Returns:

R_t = A_t + V(s_t)
Intrinsic Curiosity Module (ICM)

To encourage exploration, an Intrinsic Curiosity Module is implemented.

The ICM consists of three components:

Encoder
φ(s) = latent representation

Architecture:

obs → 256 → 128
Forward Model

Predicts the next state embedding.

F(φ(s), a) → φ(s')

Loss:

L_forward = || predicted - actual ||²
Inverse Model

Predicts the action taken.

G(φ(s), φ(s')) → action

Loss:

L_inverse = cross_entropy
Intrinsic Reward

Exploration reward is proportional to prediction error:

r_intrinsic = || F(φ(s),a) - φ(s') ||²

This encourages agents to explore novel transitions.

Reward Structure
Seeker Reward
+5  capture
-0.01 time penalty
+distance shaping
-wall collision penalty

Distance shaping:

reward += prev_dist - γ * new_dist

Encourages seekers to move toward the hider.

Hider Reward
-5 if captured
- wall collision penalty
+ survival reward

The hider learns to maximize survival time.

Curriculum Learning

Training progresses through three levels of difficulty.

Level 1
basic arena
no movable objects

Agents learn:

basic pursuit and evasion
Level 2
internal room added

Agents learn:

occlusion based hiding
search strategies
Level 3
movable box enabled

Agents can:

push objects
block visibility
create barriers

This enables emergent tool use.

Parallel Training

Training uses vectorized environments:

NUM_ENVS = 64

Each rollout collects:

BATCH_STEPS = 128

Total transitions per update:

64 × 128 = 8192

This allows efficient GPU utilization.

Training Loop

The training pipeline:

for episode:

    collect rollout
        sample actions
        step environments
        store transitions

    compute advantages

    update seeker policies
    update hider policies

    update curiosity modules

Periodic evaluation generates video recordings of learned behavior.

Emergent Behaviour Goals

The system is designed to observe emergent strategies such as:

Seekers

coordinated search

area sweeping

prediction of hiding locations

Hiders

strategic hiding

occlusion usage

object manipulation

Technologies Used

PyTorch

MuJoCo

Gymnasium

MAPPO

ICM Curiosity

AsyncVectorEnv
