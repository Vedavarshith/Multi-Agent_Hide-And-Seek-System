# Multi-Agent Hide and Seek (MAPPO + Curiosity)

This project implements a competitive **multi-agent reinforcement learning system** where two seeker agents attempt to locate a hiding agent inside a MuJoCo physics simulation. The system combines **Multi-Agent Proximal Policy Optimization (MAPPO)**, a **centralized critic**, **intrinsic curiosity driven exploration**, and **curriculum learning** to train agents operating under partial observability. The implementation is designed for **large-scale parallel rollouts**, stable physics simulation, and robust multi-agent training.

The goal of the project is to explore **emergent strategies** in competitive reinforcement learning environments where agents must learn navigation, pursuit, evasion, and environmental reasoning.

---

# 1. Environment Construction

The simulation environment is generated programmatically using **MuJoCo MJCF XML**. This allows the layout and physical components to be dynamically configured during training.

The environment consists of:

- A **20×20 bounded arena**
- Static outer boundary walls
- An **inner room structure** introducing occlusion
- A **movable box object** introduced during later curriculum stages
- Three agents:
  - Seeker 1
  - Seeker 2
  - Hider

Each agent body is constructed using:

- Two **translation slide joints** controlling movement along the X and Y axes
- One **yaw hinge joint** controlling orientation
- A forward-facing geometry indicating heading direction

Translation forces are applied using **high gear motors**, while orientation is controlled using **separate yaw motors**. This design produces simple but physically stable locomotion that maps cleanly to discrete control policies.

---

# 2. Observation Model

The environment is **partially observable**, meaning agents cannot directly access the full global state.

Instead, each agent receives a **local observation vector** consisting of:

```
[vx, vy, sin(yaw), cos(yaw), lidar_features]
```

Where:

- `vx, vy` represent the agent's local velocity
- `sin(yaw), cos(yaw)` encode the agent's orientation
- `lidar_features` provide environmental sensing information.

---

## LIDAR Sensor Model

Each agent uses a simplified **ray-based perception system** similar to LIDAR.

- **21 rays** are cast
- The rays span a **90° field of view**
- Maximum sensing distance is capped

Each ray returns:

```
(distance_normalized,
 is_hider_flag,
 is_wall_flag)
```

Thus the observation dimension becomes:

```
obs_dim = 2 velocity
        + 2 orientation
        + (21 × 3 lidar features)
        = 67
```

This allows agents to detect:

- nearby walls
- the presence of the hider
- environmental structure
- free navigable space.

---

# 3. Action Representation

Each agent performs **two discrete actions per timestep**.

### Movement

```
0 → no movement
1 → move +x
2 → move −x
3 → move +y
4 → move −y
```

### Rotation

```
0 → no rotation
1 → rotate left
2 → rotate right
```

Thus the full action for an agent is represented as:

```
(move_action, yaw_action)
```

The full environment action vector becomes:

```
[s1_move, s1_yaw,
 s2_move, s2_yaw,
 h_move,  h_yaw]
```

These discrete commands are converted into **motor torques** that drive the MuJoCo actuators.

---

# 4. Multi-Agent Learning Framework

The learning framework follows the **MAPPO (Multi-Agent PPO)** paradigm.

```
Decentralized Actors
Centralized Critic
```

Actors operate using only **local observations**, while the critic uses **global system information** during training.

This structure helps stabilize learning in a **non-stationary multi-agent environment** where each agent’s policy changes during training.

---

# 5. Actor Network

Each actor outputs two policy distributions:

```
π_move(a|s)
π_yaw(a|s)
```

The architecture is:

```
Observation
     ↓
Linear(512)
Tanh
     ↓
Linear(512)
Tanh
     ↓
Linear(256)
Tanh
     ↓
 ├─ Move Head (5 logits)
 └─ Yaw Head  (3 logits)
```

Actions are sampled using categorical distributions:

```
move ~ Categorical(move_logits)
yaw  ~ Categorical(yaw_logits)
```

Two separate actor models are trained:

```
seeker_actor
hider_actor
```

Both seekers share the same actor network parameters.

---

# 6. Centralized Critic

The critic estimates state values using **global information**.

The input to the critic is:

```
global_state =
    concat(
        obs_seeker1,
        obs_seeker2,
        obs_hider
    )
```

An **agent identity one-hot vector** is appended:

```
critic_input = concat(global_state, agent_id)
```

Critic architecture:

```
input
  ↓
Linear(512)
Tanh
  ↓
Linear(512)
Tanh
  ↓
Linear(256)
Tanh
  ↓
Linear(1)
```

This allows the critic to evaluate value functions while accounting for the full multi-agent system.

---

# 7. PPO Optimization

Policy updates follow **clipped Proximal Policy Optimization**.

Policy ratio:

```
r(θ) = πθ(a|s) / πθ_old(a|s)
```

Clipped objective:

```
L = min(
        r(θ) * A,
        clip(r(θ), 1−ε, 1+ε) * A
     )
```

Full loss:

```
L_total =
    policy_loss
  + c1 * value_loss
  - c2 * entropy_bonus
```

Where:

- value loss is mean squared error
- entropy encourages exploration
- gradients are clipped for stability.

---

# 8. Advantage Estimation

Advantages are computed using **Generalized Advantage Estimation (GAE)**.

Temporal difference error:

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
```

Recursive advantage:

```
A_t = δ_t + γλ A_{t+1}
```

Returns are computed as:

```
R_t = A_t + V(s_t)
```

Advantages are normalized before performing PPO updates.

---

# 9. Intrinsic Curiosity Module (ICM)

To encourage exploration, the system integrates **Intrinsic Curiosity Modules**.

Each module consists of:

- an **encoder**
- a **forward dynamics model**
- an **inverse dynamics model**

---

## Encoder

Encodes observations into latent features.

```
obs → Linear(256) → ReLU → Linear(128)
```

Output:

```
φ(s)
```

---

## Forward Model

Predicts the next latent state.

```
F(φ(s), a) → φ(s')
```

Loss:

```
L_forward = || predicted − actual ||²
```

---

## Inverse Model

Predicts the action taken between two states.

```
G(φ(s), φ(s')) → action
```

Loss:

```
L_inverse = CrossEntropy
```

---

## Intrinsic Reward

Curiosity reward is proportional to forward prediction error:

```
r_intrinsic =
    ||F(φ(s), a) − φ(s')||²
```

Rewards are normalized using a **running mean and variance estimator** before being added to extrinsic rewards.

---

# 10. Reward Design

### Seeker Reward

Seekers are incentivized to find the hider.

```
+5   capture reward
−0.01 step penalty
+distance shaping
−wall collision penalty
```

Distance shaping:

```
reward += prev_distance − γ * current_distance
```

---

### Hider Reward

The hider aims to maximize survival time.

```
−5 capture penalty
−wall collision penalty
+survival reward
```

---

# 11. Curriculum Learning

Training progresses through **three difficulty levels**.

### Level 1

```
Open arena
No movable objects
```

Agents learn basic:

```
navigation
pursuit
evasion
```

---

### Level 2

```
Internal room added
```

Agents learn:

```
occlusion reasoning
structured search
```

---

### Level 3

```
Movable box enabled
```

Agents can manipulate objects to:

```
block visibility
create barriers
```

This stage enables **emergent tool-use behaviors**.

---

# 12. Parallel Training

Training uses **vectorized environments**.

```
NUM_ENVS = 64
BATCH_STEPS = 128
```

Transitions per update:

```
64 × 128 = 8192
```

Parallel rollout collection significantly improves training throughput.

---

# 13. Training Pipeline

The training loop proceeds as:

```
for episode:

    collect rollout across environments

    sample actions from actor policies

    execute environment steps

    compute intrinsic curiosity rewards

    store transitions

    compute GAE advantages

    perform PPO updates

    update curiosity models
```

Evaluation runs periodically generate **videos of learned behaviors**.

---

# 14. Engineering Considerations

The implementation includes several robustness improvements:

- MuJoCo renderer destructor patch to prevent Python 3.12 crashes
- actuator name mapping for stable control indexing
- fallback control logic for inconsistent model layouts
- safe vectorized environment curriculum updates
- velocity clipping for physics stability

These design choices ensure **stable large-scale multi-agent training**.

---

# Technologies Used

- Python
- PyTorch
- MuJoCo
- Gymnasium
- MAPPO
- Intrinsic Curiosity Module
- AsyncVectorEnv

---

# License

MIT License
