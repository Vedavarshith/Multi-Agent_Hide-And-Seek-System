# Full updated script with robust set_level and corrected Level-1 layout
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium.vector as vector
import mujoco
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import imageio
import gc
import warnings
import time

# ---- HARD PATCH: Fix MuJoCo Renderer destructor crash (Python 3.12 / EGL) ----
import mujoco.renderer

def _safe_renderer_del(self):
    try:
        if hasattr(self, "_mjr_context") and self._mjr_context:
            self.close()
    except Exception:
        pass

mujoco.renderer.Renderer.__del__ = _safe_renderer_del

# --- HARDWARE / ENV SETUP ---
torch.set_float32_matmul_precision('high')
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 HARDWARE DETECTED: {device}")

# =====================
# 1) XML generation (REVISED layout: opening facing arena center)
# =====================
def generate_curriculum_xml():
    """
    MJCF generator:
    - translation slide joints for x,y for each agent
    - nested child body per agent with a hinge yaw joint and a forward-facing geom
    - yaw motors added to actuators (so yaw is actuated)
    """
    xml_content = f"""
<mujoco model="HideAndSeek_Curriculum">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option timestep="0.005" gravity="0 0 -9.81" density="1.2" viscosity="0.0"/>
    <visual>
        <global offwidth="1920" offheight="1080" fovy="45"/>
        <quality shadowsize="0"/>
        <map fogstart="40" fogend="80"/>
    </visual>
    <default>
        <joint armature="0" damping="0.1" limited="true"/>
        <geom conaffinity="1" condim="3" density="5.0" friction="0.1 0.1 0.1" margin="0.01"/>
    </default>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.9 0.92 0.95" rgb2="1.0 1.0 1.0" width="512" height="512"/>
        <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3" width="1024" height="1024"/>
        <material name="mat_floor" texture="tex_floor" reflectance="0.8" shininess="1.0" specular="1.0" texrepeat="8 8"/>
        <material name="mat_wall" rgba="0.95 0.95 0.95 1" reflectance="0.1" shininess="0.2"/>
        <material name="mat_seeker" rgba="0.2 0.6 1.0 1"/>
        <material name="mat_hider" rgba="0.9 0.1 0.1 1"/>
        <material name="mat_box" rgba="0.6 0.4 0.2 1"/>
    </asset>

    <worldbody>
        <light diffuse="0.85 0.85 0.88" dir="-0.3 0.3 -1" pos="0 0 20" directional="true"/>
        <camera name="closeup" pos="0 -26 24" xyaxes="1 0 0 0 0.6 0.8"/>

        <geom name="floor" material="mat_floor" pos="0 0 0" size="15 15 0.1" type="plane"/>

        <body name="walls_static" pos="0 0 0">
            <geom name="wall_top" material="mat_wall" pos="0 10.0 0.5" size="10.5 0.5 0.8" type="box"/>
            <geom name="wall_btm" material="mat_wall" pos="0 -10.0 0.5" size="10.5 0.5 0.8" type="box"/>
            <geom name="wall_lft" material="mat_wall" pos="-10.0 0 0.5" size="0.5 10.5 0.8" type="box"/>
            <geom name="wall_rgt" material="mat_wall" pos="10.0 0 0.5" size="0.5 10.5 0.8" type="box"/>
        </body>

        <body name="inner_room" pos="5 -5 0.5">
            <geom name="room_wall_r" material="mat_wall" pos="3 0 0" size="0.2 3.2 0.8" type="box"/>
            <geom name="room_wall_b" material="mat_wall" pos="0 -3 0" size="3.2 0.2 0.8" type="box"/>
            <geom name="room_wall_l" material="mat_wall" pos="-3 0 0" size="0.2 3.2 0.8" type="box"/>
            <geom name="room_door_l" material="mat_wall" pos="-2 3 0" size="1.2 0.2 0.8" type="box"/>
            <geom name="room_door_r" material="mat_wall" pos="2 3 0" size="1.2 0.2 0.8" type="box"/>
        </body>

        <!-- seeker1: slide x,y in outer body, nested rot body with hinge + forward geom for heading -->
        <body name="seeker1" pos="-3 0 0.5">
            <joint name="s1_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="s1_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <body name="seeker1_rot" pos="0 0 0">
                <joint name="s1_yaw" type="hinge" axis="0 0 1" range="-180 180"/>
                <geom name="geom_seeker1" material="mat_seeker" type="box" pos="0.6 0 0" size="0.25 0.6 0.25"/>
            </body>
        </body>

        <!-- seeker2 -->
        <body name="seeker2" pos="-3 2 0.5">
            <joint name="s2_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="s2_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <body name="seeker2_rot" pos="0 0 0">
                <joint name="s2_yaw" type="hinge" axis="0 0 1" range="-180 180"/>
                <geom name="geom_seeker2" material="mat_seeker" type="box" pos="0.6 0 0" size="0.25 0.6 0.25"/>
            </body>
        </body>

        <!-- hider -->
        <body name="hider" pos="5 -5 0.5">
            <joint name="h_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="h_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <body name="hider_rot" pos="0 0 0">
                <joint name="h_yaw" type="hinge" axis="0 0 1" range="-180 180"/>
                <geom name="geom_hider" material="mat_hider" type="box" pos="0.6 0 0" size="0.25 0.6 0.25"/>
            </body>
        </body>

        <!-- moveable box -->
        <body name="moveable_box" pos="0 0 0.5">
            <joint name="box_x" type="slide" axis="1 0 0" range="-9.0 9.0"/>
            <joint name="box_y" type="slide" axis="0 1 0" range="-9.0 9.0"/>
            <geom name="geom_box" material="mat_box" size="0.8 0.8 0.5" type="box" mass="2.0"/>
        </body>

    </worldbody>

    <actuator>
        <motor name="thrust_s1_x" joint="s1_x" gear="10000"/>
        <motor name="thrust_s1_y" joint="s1_y" gear="10000"/>
        <motor name="thrust_s2_x" joint="s2_x" gear="10000"/>
        <motor name="thrust_s2_y" joint="s2_y" gear="10000"/>
        <motor name="thrust_h_x" joint="h_x" gear="10000"/>
        <motor name="thrust_h_y" joint="h_y" gear="10000"/>
        <!-- yaw motors (actuated yaw control) -->
        <motor name="yaw_s1" joint="s1_yaw" gear="800"/>
        <motor name="yaw_s2" joint="s2_yaw" gear="800"/>
        <motor name="yaw_h"  joint="h_yaw"  gear="800"/>
    </actuator>
</mujoco>
"""
    return xml_content


# write xml
with open("mappo_curriculum.xml", "w") as f:
    f.write(generate_curriculum_xml())


# =====================
# 2) CurriculumEnv (same as before except potential shaping kept; no change)
# =====================

class CurriculumEnv(gym.Env):
    """
    Curriculum environment implementing:
      - actuated yaw per agent
      - limited FOV raycasts (n_rays over fov_deg)
      - per-agent obs: [vx, vy, sin(yaw), cos(yaw), (d_norm,is_hider,is_wall)*n_rays]
      - action_space: MultiDiscrete([5,3,5,3,5,3]) => (move,yaw)*3 agents
    """
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = None

        self.level = 1

        # FOV / LIDAR
        self.fov_deg = 90.0
        self.n_rays = 21                 # odd preferred (center ray)
        self.max_sight = 15.0
        self.fov = np.deg2rad(self.fov_deg)
        self.ray_angles = np.linspace(-self.fov/2, self.fov/2, self.n_rays)

        # ACTIONS: for each agent -> (move 5-way, yaw 3-way)
        # move: 0=noop,1:+x,2:-x,3:+y,4:-y
        # yaw:  0=noop,1=rotate_left,2=rotate_right
        self.action_space = spaces.MultiDiscrete([5, 3, 5, 3, 5, 3])

        # OBS: vel_x, vel_y, sin(yaw), cos(yaw), (d_norm,is_hider,is_wall)*n_rays
        self.obs_dim = 2 + 2 + (self.n_rays * 3)

        max_vel = 20.0
        lidar_high = np.ones(self.n_rays * 3, dtype=np.float32) * self.max_sight
        high_row = np.concatenate([np.array([max_vel, max_vel, 1.0, 1.0], dtype=np.float32), lidar_high]).astype(np.float32)
        high = np.tile(high_row, (3, 1))
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # cache geom/body ids
        try:
            self.id_hider = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "geom_hider")
        except Exception:
            self.id_hider = -1

        self.wall_geom_ids = []
        for n in ["wall_top","wall_btm","wall_lft","wall_rgt","room_wall_r","room_wall_b","room_wall_l"]:
            try:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                if gid >= 0:
                    self.wall_geom_ids.append(gid)
            except Exception:
                pass

        try:
            self.id_box = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box")
        except Exception:
            self.id_box = -1

        self.agent_bodies = ["seeker1", "seeker2", "hider"]
        self.agent_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.agent_bodies]

        # find yaw joint qpos addresses (for reading yaw) by joint name
        self.yaw_jnames = ["s1_yaw", "s2_yaw", "h_yaw"]
        self.yaw_qpos_idxs = []
        for jn in self.yaw_jnames:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                adr = int(self.model.jnt_qposadr[jid])
                self.yaw_qpos_idxs.append(adr)
            except Exception:
                self.yaw_qpos_idxs.append(-1)

        # map actuator names -> indices (for setting yaw motor controls robustly)
        self.actuator_name_to_idx = {}
        try:
            for i in range(self.model.nu):
                try:
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).decode()
                except Exception:
                    try:
                        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    except Exception:
                        name = None
                if name:
                    self.actuator_name_to_idx[name] = i
        except Exception:
            # best-effort mapping; will still work if actuator names are default order
            self.actuator_name_to_idx = {}

        # store last yaw (fallback when qpos indexing is not available)
        self.yaws = np.zeros(3, dtype=np.float64)

        # find body qpos index for moveable box (optional)
        self.box_qpos_idx = -1
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moveable_box")
            for j in range(self.model.njnt):
                if int(self.model.jnt_bodyid[j]) == int(body_id):
                    adr = int(self.model.jnt_qposadr[j])
                    if adr >= 0:
                        self.box_qpos_idx = adr
                        break
        except Exception:
            self.box_qpos_idx = -1

        self.prev_min_dist = None

    def set_level(self, level):
        self.level = level

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # level-specific object placement
        barrier_z = -100.0 if self.level == 1 else 0.5
        box_z = 0.5 if self.level >= 3 else -100.0

        if hasattr(self, "id_barrier") and getattr(self, "id_barrier", -1) >= 0:
            self.model.body_pos[self.id_barrier][2] = barrier_z
        if self.id_box >= 0:
            self.model.body_pos[self.id_box][2] = box_z

        # Set top-level body positions for agents (robust to qpos layout)
        # seeker1, seeker2, hider positions randomized (z kept 0.5)
        try:
            bid_s1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seeker1")
            bid_s2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seeker2")
            bid_h  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hider")
            self.model.body_pos[bid_s1][:2] = np.array([np.random.uniform(-8, 8), np.random.uniform(-8, -2)])
            self.model.body_pos[bid_s2][:2] = np.array([np.random.uniform(-8, 8), np.random.uniform(-8, -2)])
            self.model.body_pos[bid_h][:2]  = np.array([np.random.uniform(-8, 8), np.random.uniform(2, 8)])
        except Exception:
            # fallback: try qpos randomization similar to earlier code
            try:
                self.data.qpos[0] = np.random.uniform(-8, 8)
                self.data.qpos[1] = np.random.uniform(-8, -2)
                self.data.qpos[2] = np.random.uniform(-8, 8)
                self.data.qpos[3] = np.random.uniform(-8, -2)
                self.data.qpos[4] = np.random.uniform(-8, 8)
                self.data.qpos[5] = np.random.uniform(2, 8)
            except Exception:
                pass

        # optional box placement
        if self.level >= 3:
            valid_pos = False
            while not valid_pos:
                bx, by = np.random.uniform(-5, 5), np.random.uniform(-3, 3)
                if not ((bx > -4.0 and bx < 4.0) and (by > -1.5 and by < 1.5)):
                    valid_pos = True
                    if self.box_qpos_idx >= 0:
                        self.data.qpos[self.box_qpos_idx] = bx
                        self.data.qpos[self.box_qpos_idx+1] = by
                    else:
                        if self.id_box >= 0:
                            self.model.body_pos[self.id_box][0] = bx
                            self.model.body_pos[self.id_box][1] = by

        mujoco.mj_forward(self.model, self.data)

        # set initial yaw to face arena center (0,0)
        for i, bname in enumerate(self.agent_bodies):
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
                pos = np.array(self.data.body_xpos[bid][:2])
                yaw = float(np.arctan2(-pos[1], -pos[0]))
                self.yaws[i] = yaw
                if self.yaw_qpos_idxs[i] >= 0:
                    self.data.qpos[self.yaw_qpos_idxs[i]] = yaw
            except Exception:
                # ignore if any lookup fails
                pass

        mujoco.mj_forward(self.model, self.data)

        # compute previous distances robustly using body positions
        try:
            s1p = np.array(self.data.body_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seeker1")][:2])
            s2p = np.array(self.data.body_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seeker2")][:2])
            hpos = np.array(self.data.body_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hider")][:2])
            self.prev_min_dist = float(min(np.linalg.norm(s1p - hpos), np.linalg.norm(s2p - hpos)))
        except Exception:
            self.prev_min_dist = None

        return self._get_obs(), {}

    def step(self, actions):
        """
        actions: sequence-like of length 6 (move0,yaw0, move1,yaw1, move2,yaw2)
        Each element is an integer per MultiDiscrete specification.
        """
        # normalize actions to list of ints
        try:
            acts = [int(x) for x in actions]
        except Exception:
            acts = list(map(int, np.asarray(actions).ravel()))

        # zero controls
        self.data.ctrl[:] = 0.0

        # translation mapping (controls arranged as 0..5 for translation motors)
        # we assume translation motors are in consistent order; set by index mapping or by manual base as earlier
        # fallback to direct ctrl indexing by base = i*2 if actuator name mapping absent
        # first try to set translation by actuator names if present
        translation_names = ['thrust_s1_x', 'thrust_s1_y', 'thrust_s2_x', 'thrust_s2_y', 'thrust_h_x', 'thrust_h_y']
        translation_set = False
        try:
            # if all translation names exist in mapping, use them
            if all(n in self.actuator_name_to_idx for n in translation_names):
                for i_agent in range(3):
                    move = acts[i_agent*2]
                    idx_x = self.actuator_name_to_idx[translation_names[i_agent*2]]
                    idx_y = self.actuator_name_to_idx[translation_names[i_agent*2 + 1]]
                    if move == 1:
                        self.data.ctrl[idx_x] = 1.0
                    elif move == 2:
                        self.data.ctrl[idx_x] = -1.0
                    elif move == 3:
                        self.data.ctrl[idx_y] = 1.0
                    elif move == 4:
                        self.data.ctrl[idx_y] = -1.0
                translation_set = True
        except Exception:
            translation_set = False

        if not translation_set:
            # fallback to base indexing (assumes first 6 ctrl indices are translation)
            for i_agent in range(3):
                base = i_agent * 2
                move = acts[i_agent*2]
                if move == 1: self.data.ctrl[base] = 1.0
                elif move == 2: self.data.ctrl[base] = -1.0
                elif move == 3: self.data.ctrl[base+1] = 1.0
                elif move == 4: self.data.ctrl[base+1] = -1.0

        # yaw controls mapping (discrete -> torque)
        yaw_torque_map = {0: 0.0, 1: +1.0, 2: -1.0}
        yaw_scale = 0.5   # tune this in experiments

        yaw_actuator_names = ['yaw_s1', 'yaw_s2', 'yaw_h']
        for i_agent in range(3):
            yaw_act = acts[i_agent*2 + 1]
            name = yaw_actuator_names[i_agent]
            if name in self.actuator_name_to_idx:
                idx = self.actuator_name_to_idx[name]
                self.data.ctrl[idx] = float(yaw_scale * yaw_torque_map.get(int(yaw_act), 0.0))
            else:
                # fallback: if actuator mapping missing, attempt to set qvel of yaw joint directly (not preferred)
                try:
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.yaw_jnames[i_agent])
                    if jid >= 0:
                        # apply small delta to qvel
                        self.data.qvel[self.model.jnt_qveladr[jid]] = float(yaw_scale * yaw_torque_map.get(int(yaw_act), 0.0))
                except Exception:
                    pass

        # clip velocities and step
        self.data.qvel[:] = np.clip(self.data.qvel, -15.0, 15.0)
        try:
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
        except Exception:
            # on stepping failure, return zeroed obs & terminal
            return np.zeros((3, self.obs_dim), dtype=np.float32), np.zeros(2, dtype=np.float32), True, False, {}

        # forward to update internals (positions, qpos, etc.)
        mujoco.mj_forward(self.model, self.data)

        # ensure our yaw reading fallback updated
        for i in range(3):
            if self.yaw_qpos_idxs[i] >= 0:
                try:
                    self.yaws[i] = float(self.data.qpos[self.yaw_qpos_idxs[i]])
                except Exception:
                    pass

        # build observation
        obs = self._get_obs()

        # Determine team visibility using lidar 'is_hider' flags from obs
        seen_flags = []
        for i in range(3):
            lidar_flat = obs[i, 2:]
            is_hider_flags = lidar_flat[1::3]
            seen_flags.append(np.any(is_hider_flags > 0.5))
        team_sees = bool(seen_flags[0] or seen_flags[1])

        # distances for shaping (use body positions)
        try:
            s1p = np.array(self.data.body_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seeker1")][:2])
            s2p = np.array(self.data.body_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "seeker2")][:2])
            hpos = np.array(self.data.body_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hider")][:2])
            dist_s1 = float(np.linalg.norm(s1p - hpos))
            dist_s2 = float(np.linalg.norm(s2p - hpos))
            curr_min_dist = min(dist_s1, dist_s2)
        except Exception:
            curr_min_dist = None

        # shaping and rewards (kept consistent with your spec)
        gamma_shaping = 0.99
        shape_scale = 1.0
        if self.prev_min_dist is None or curr_min_dist is None:
            shaping = 0.0
        else:
            shaping = float(self.prev_min_dist - gamma_shaping * curr_min_dist) * shape_scale

        # per-step extrinsic reward design:
        # seeker: r = 1.0 if team sees hider else small negative step
        # hider: r = (1 - seen) - alpha * seen
        r_step = -0.01
        capture_thresh = 0.6
        if curr_min_dist is not None and curr_min_dist < capture_thresh:
            r_seek = 5.0 + r_step
            r_hide = -5.0
        else:
            r_seek = r_step
            r_hide = 0.0

        r_seek += shaping

        # wall penalty using lidar is_wall flag
        def wall_hit(lidar_flat):
            dists = lidar_flat[0::3]
            wall_flags = lidar_flat[2::3]
            return np.any((dists < 0.15) & (wall_flags > 0.5))

        if wall_hit(obs[0, 2:]) or wall_hit(obs[1, 2:]):
            r_seek -= 0.2
        if wall_hit(obs[2, 2:]):
            r_hide -= 0.2

        if curr_min_dist is not None:
            self.prev_min_dist = curr_min_dist

        return obs, np.array([r_seek, r_hide], dtype=np.float32), False, False, {}

    def _get_obs(self):
        """
        Returns array shape (3, obs_dim) per-agent observations:
        [vx, vy, sin(yaw), cos(yaw), (d_norm,is_hider,is_wall)*n_rays]
        Rays are cast relative to current yaw for each agent.
        """
        observations = []
        for i in range(3):
            # body id and base info
            try:
                bid = self.agent_body_ids[i]
                pos = np.array(self.data.body_xpos[bid][:2])
                vel = np.array(self.data.body_xvel[bid][:2])
            except Exception:
                # fallback to qpos/qvel slices (best-effort)
                base_idx = i * 2
                pos = np.array(self.data.qpos[base_idx:base_idx+2]) if self.data.qpos.size > base_idx+1 else np.zeros(2)
                vel = np.array(self.data.qvel[base_idx:base_idx+2]) if self.data.qvel.size > base_idx+1 else np.zeros(2)

            # yaw read (prefer qpos adr)
            if self.yaw_qpos_idxs[i] >= 0:
                try:
                    yaw = float(self.data.qpos[self.yaw_qpos_idxs[i]])
                except Exception:
                    yaw = float(self.yaws[i])
            else:
                yaw = float(self.yaws[i])

            yaw_sin = float(np.sin(yaw))
            yaw_cos = float(np.cos(yaw))

            lidar_data = []
            start_pos = np.array([pos[0], pos[1], 0.5], dtype=np.float64)

            for r_idx, ang in enumerate(self.ray_angles):
                # rotate the ray by yaw
                global_angle = yaw + ang
                vec = np.array([np.cos(global_angle), np.sin(global_angle), 0.0], dtype=np.float64)

                geomid = np.zeros(1, dtype=np.int32)
                # mj_ray returns distance or -1 on miss depending on API
                try:
                    d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, int(self.agent_body_ids[i]), geomid)
                except Exception:
                    # some mujoco builds accept slightly different args; best-effort simple call:
                    try:
                        d = mujoco.mj_ray(self.model, self.data, start_pos, vec, None, 1, geomid)
                    except Exception:
                        d = -1

                g = int(geomid[0]) if geomid.size > 0 else -1

                is_hider = 0.0
                is_wall = 0.0
                if d == -1 or d is None or d > self.max_sight:
                    d_val = self.max_sight
                else:
                    d_val = float(d)
                    if self.id_hider >= 0 and g == int(self.id_hider):
                        is_hider = 1.0
                    elif g in [int(x) for x in self.wall_geom_ids]:
                        is_wall = 1.0

                lidar_data.extend([d_val / self.max_sight, is_hider, is_wall])

            obs_vec = np.concatenate([vel, np.array([yaw_sin, yaw_cos], dtype=np.float32), np.array(lidar_data, dtype=np.float32)])
            observations.append(obs_vec)

        return np.stack(observations).astype(np.float32)

    def render(self):
        # same robust renderer as before
        if getattr(self, "_renderer_ok", True) is False:
            return np.zeros((240,320,3), dtype=np.uint8)
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.model, height=240, width=320)
            except Exception:
                self.renderer = None
                self._renderer_ok = False
                return np.zeros((240,320,3), dtype=np.uint8)
        try:
            self.renderer.update_scene(self.data, camera="closeup")
            frame = self.renderer.render()
            return frame
        except Exception:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
            self._renderer_ok = False
            return np.zeros((240,320,3), dtype=np.uint8)

    def close(self):
        if self.renderer is None:
            return
        try:
            self.renderer.close()
        except Exception:
            pass
        finally:
            self.renderer = None
            self._renderer_ok = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

# =====================
# 3) Networks and ICM + optimizer code (same as previous final)
# =====================
class Actor(nn.Module):
    """
    Shared backbone with two discrete action heads:
      - move_head: 5-way (no-op, +x, -x, +y, -y)
      - yaw_head: 3-way (no-op, left, right)
    Forward returns logits for both heads (preferred for Categorical with logits).
    """
    def __init__(self, obs_dim, move_dim=5, yaw_dim=3, hidden=512):
        super().__init__()
        self.move_dim = move_dim
        self.yaw_dim = yaw_dim

        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden//2), nn.Tanh()
        )
        self.move_head = nn.Linear(hidden//2, move_dim)   # logits
        self.yaw_head  = nn.Linear(hidden//2, yaw_dim)    # logits

    def forward(self, x):
        # x: [B, obs_dim]
        feat = self.base(x)
        move_logits = self.move_head(feat)
        yaw_logits = self.yaw_head(feat)
        return move_logits, yaw_logits


class CentralizedCritic(nn.Module):
    def __init__(self, global_obs_dim, n_agents=3):
        super().__init__()
        self.n_agents = n_agents
        in_dim = global_obs_dim + n_agents
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
    def forward(self, global_obs, agent_ids):
        device = global_obs.device
        b = global_obs.size(0)
        onehot = torch.zeros(b, self.n_agents, device=device)
        onehot.scatter_(1, agent_ids.unsqueeze(1), 1.0)
        inp = torch.cat([global_obs, onehot], dim=1)
        return self.net(inp)

class ICM(nn.Module):
    """
    ICM encoder + forward/inverse models.
    Action representation: concatenated one-hot of (move_dim + yaw_dim).
    """

    def __init__(self, obs_dim, move_dim=5, yaw_dim=3, forward_coef=1.0, inverse_coef=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.move_dim = move_dim
        self.yaw_dim = yaw_dim
        self.act_dim = move_dim + yaw_dim
        self.forward_coef = forward_coef
        self.inverse_coef = inverse_coef

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(128 + self.act_dim, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(128 + 128, 256), nn.ReLU(),
            nn.Linear(256, self.move_dim)  # inverse predicts *move* only (or joint mapping; here move chosen)
        )

    def _make_action_onehot(self, move_idx, yaw_idx):
        # move_idx, yaw_idx: Long tensors shape (B,)
        batch = move_idx.size(0)
        act_onehot = torch.zeros(batch, self.act_dim, device=move_idx.device)
        act_onehot.scatter_(1, move_idx.unsqueeze(1).long(), 1.0)
        # yaw placed after move_dim
        yaw_onehot = torch.zeros(batch, self.yaw_dim, device=move_idx.device)
        yaw_onehot.scatter_(1, yaw_idx.unsqueeze(1).long(), 1.0)
        act_onehot[:, self.move_dim:] = yaw_onehot
        return act_onehot

    def compute_intrinsic_reward(self, state, next_state, move_idx, yaw_idx):
        """
        state, next_state: [B, obs_dim]
        move_idx, yaw_idx: [B] integer tensors
        returns per-sample forward error (detached)
        """
        curr = self.encoder(state)
        next_enc = self.encoder(next_state)
        act_onehot = self._make_action_onehot(move_idx, yaw_idx)
        pred = self.forward_model(torch.cat([curr, act_onehot], dim=1))
        fwd_error = 0.5 * (pred - next_enc).pow(2).mean(dim=1)
        return fwd_error.detach()

    def compute_loss(self, state, next_state, move_idx, yaw_idx):
        curr = self.encoder(state)
        next_enc = self.encoder(next_state)
        act_onehot = self._make_action_onehot(move_idx, yaw_idx)

        pred_next = self.forward_model(torch.cat([curr, act_onehot], dim=1))
        forward_loss = 0.5 * (pred_next - next_enc).pow(2).mean()

        inv_logits = self.inverse_model(torch.cat([curr, next_enc], dim=1))
        # inverse targets: we can train to predict move_idx (discrete), which is typical
        inverse_loss = F.cross_entropy(inv_logits, move_idx.long())

        loss = self.forward_coef * forward_loss + self.inverse_coef * inverse_loss
        return loss, forward_loss.detach()

# GAE, PPO update, normalizer etc (same as earlier final)...
def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    gae = torch.zeros_like(next_value)
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[step])
        next_value = values[step]
    returns = torch.stack(returns)
    advantages = returns - torch.stack(values)
    return returns, advantages

def ppo_update(actor, critic, optimizer,
               states, actions_move, actions_yaw, old_log_probs, returns, advantages,
               global_states, agent_ids,
               epochs=10, minibatch_size=4096, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
    """
    actions_move, actions_yaw: Long tensors shape [B]
    old_log_probs: tensor [B] (sum of old move+yaw logprobs)
    """
    BATCH_SIZE = states.size(0)
    device = states.device
    all_params = list(actor.parameters()) + list(critic.parameters())

    for _ in range(epochs):
        indices = torch.randperm(BATCH_SIZE, device=device)
        for start in range(0, BATCH_SIZE, minibatch_size):
            end = min(start + minibatch_size, BATCH_SIZE)
            idx = indices[start:end]
            if idx.numel() == 0:
                continue

            mb_states = states[idx]
            mb_actions_move = actions_move[idx].long()
            mb_actions_yaw  = actions_yaw[idx].long()
            mb_old_lp = old_log_probs[idx]
            mb_returns = returns[idx]
            mb_adv = advantages[idx]
            mb_glob = global_states[idx]
            mb_agent_ids = agent_ids[idx]

            move_logits, yaw_logits = actor(mb_states)
            dist_move = Categorical(logits=move_logits)
            dist_yaw  = Categorical(logits=yaw_logits)

            new_lp_move = dist_move.log_prob(mb_actions_move)
            new_lp_yaw  = dist_yaw.log_prob(mb_actions_yaw)
            new_log_probs = new_lp_move + new_lp_yaw
            entropy = dist_move.entropy().mean() + dist_yaw.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            values = critic(mb_glob, mb_agent_ids).squeeze()
            value_loss = 0.5 * (mb_returns - values).pow(2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
            optimizer.step()

class RunningIntrinsicNormalizer:
    def __init__(self, momentum=0.01, eps=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.momentum = momentum
        self.eps = eps
        self.inited = False

    def update_and_normalize(self, x_tensor):
        x = x_tensor.detach()
        batch_mean = float(x.mean().item()) if x.numel()>0 else 0.0
        batch_var = float(x.var(unbiased=False).item()) if x.numel()>1 else 0.0
        if not self.inited:
            self.mean = batch_mean
            self.var = batch_var if batch_var > 0 else 1.0
            self.inited = True
        else:
            m = self.momentum
            self.mean = (1 - m) * self.mean + m * batch_mean
            self.var = (1 - m) * self.var + m * batch_var
        std = (self.var ** 0.5) + self.eps
        return (x - self.mean) / std

# =====================
# 4) Training loop + robust set_level helper
# =====================
def make_env(xml_path):
    def _env(): return CurriculumEnv(xml_path)
    return _env

def set_vector_env_level(envs, level, xml_path, make_env_fn, num_envs):
    """
    Try several vector APIs to set level. If none available, recreate envs.
    Returns (envs, recreated_flag)
    """
    # try .env_method
    try:
        envs.env_method("set_level", level)
        print("[set_level] used env_method")
        return envs, False
    except Exception:
        pass
    # try .set_attr
    try:
        envs.set_attr("level", level)
        print("[set_level] used set_attr")
        return envs, False
    except Exception:
        pass
    # try .call (older gymnasium vector API)
    try:
        envs.call("set_level", level)
        print("[set_level] used call()")
        return envs, False
    except Exception:
        pass
    # fallback: try to access internal env list (if exists)
    try:
        if hasattr(envs, "envs") and envs.envs is not None:
            for e in envs.envs:
                try:
                    e.set_level(level)
                except Exception:
                    pass
            print("[set_level] used envs list")
            return envs, False
    except Exception:
        pass

    # Last resort: recreate vector envs (safe but heavyweight)
    print("[set_level] Couldn't set level via vector API — recreating vector envs to apply level.")
    try:
        envs.close()
    except Exception:
        pass

    new_envs = vector.AsyncVectorEnv([make_env_fn(xml_path) for _ in range(num_envs)])
    # try to set level on new_envs using same attempts
    try:
        new_envs.env_method("set_level", level)
    except Exception:
        try:
            new_envs.set_attr("level", level)
        except Exception:
            try:
                new_envs.call("set_level", level)
            except Exception:
                # if still can't, we'll rely on eval_env.set_level and hope new envs start default and then use reset to apply
                print("[set_level] recreated envs but couldn't set via API; ensure environments read level on reset or call reset manually.")
    return new_envs, True

if __name__ == "__main__":
    gc.collect()
    xml_path = "mappo_curriculum.xml"

    NUM_ENVS = 64
    print(f"⚡ Launching {NUM_ENVS} Parallel Environments...")
    envs = vector.AsyncVectorEnv([make_env(xml_path) for _ in range(NUM_ENVS)])
    eval_env = CurriculumEnv(xml_path)

    obs_dim = eval_env.obs_dim
    global_obs_dim = obs_dim * 3

    seeker_actor = Actor(obs_dim, move_dim=5, yaw_dim=3).to(device)
    seeker_critic = CentralizedCritic(global_obs_dim, n_agents=3).to(device)
    hider_actor   = Actor(obs_dim, move_dim=5, yaw_dim=3).to(device)
    hider_critic = CentralizedCritic(global_obs_dim, n_agents=3).to(device)

    icm_seekers = ICM(obs_dim, move_dim=5, yaw_dim=3).to(device)
    icm_hider   = ICM(obs_dim, move_dim=5, yaw_dim=3).to(device)


    opt_s = optim.Adam(list(seeker_actor.parameters()) + list(seeker_critic.parameters()), lr=3e-4)
    opt_h = optim.Adam(list(hider_actor.parameters()) + list(hider_critic.parameters()), lr=3e-4)
    opt_icm_s = optim.Adam(icm_seekers.parameters(), lr=1e-3)
    opt_icm_h = optim.Adam(icm_hider.parameters(), lr=1e-3)

    icm_norm_seekers = RunningIntrinsicNormalizer(momentum=0.01)
    icm_norm_hider = RunningIntrinsicNormalizer(momentum=0.01)

    EPISODES = 1000
    BATCH_STEPS = 128
    current_level = 1

    icm_scale = 0.1
    icm_scale_h = 0.06

    # Precompute agent id vectors for seekers ordering (s1 then s2)
    # Note: we'll build full-length agent id arrays per update as needed (they must match flattened batch size)
    agent_ids_seekers = None  # built per update
    agent_id_hider = None

    for episode in range(EPISODES):
        start_time = time.time()

        # Level progression logic
        target = 3 if episode >= 350 else (2 if episode >= 150 else 1)
        if target > current_level:
            current_level = target
            # Attempt to set level across vectorized envs robustly
            envs, recreated = set_vector_env_level(envs, current_level, xml_path, make_env, NUM_ENVS)
            if recreated:
                # after recreation it's safe to reset to ensure state consistent
                obs_batch, _ = envs.reset()
            eval_env.set_level(current_level)
            print(f"[main] Level updated to {current_level} (recreated={recreated})")

        # Recording logic unchanged (kept as requested)
        if (episode % 100 == 99) or (episode == 349) or (episode == 149):
            print(f"🎥 RECORDING EPISODE {episode}...")
            video_obs, _ = eval_env.reset()
            filename = f'l40s_mappo_lvl{current_level}_ep{episode}.mp4'

            with imageio.get_writer(filename, fps=30) as writer:
                for t in range(1000):
                    try:
                        frame = eval_env.render()
                        writer.append_data(frame)
                    except: pass

                    obs_t_vid = torch.tensor(video_obs, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        # seeker1
                        mlog_s1, ylog_s1 = seeker_actor(obs_t_vid[0].unsqueeze(0))
                        s1_move = int(torch.argmax(mlog_s1, dim=-1).item())
                        s1_yaw  = int(torch.argmax(ylog_s1, dim=-1).item())

                        # seeker2
                        mlog_s2, ylog_s2 = seeker_actor(obs_t_vid[1].unsqueeze(0))
                        s2_move = int(torch.argmax(mlog_s2, dim=-1).item())
                        s2_yaw  = int(torch.argmax(ylog_s2, dim=-1).item())

                        # hider
                        mlog_h, ylog_h = hider_actor(obs_t_vid[2].unsqueeze(0))
                        h_move = int(torch.argmax(mlog_h, dim=-1).item())
                        h_yaw  = int(torch.argmax(ylog_h, dim=-1).item())

                    video_obs, _, _, _, _ = eval_env.step([s1_move, s1_yaw, s2_move, s2_yaw, h_move, h_yaw])

            gc.collect()
            print(f"💾 Saved {filename}")

        # Training buffers (reset per episode)
        b_s_obs, b_s_act_move, b_s_act_yaw, b_s_lp, b_s_val, b_s_rew = [], [], [], [], [], []
        b_h_obs, b_h_act_move, b_h_act_yaw, b_h_lp, b_h_val, b_h_rew = [], [], [], [], [], []
        b_glob_s, b_glob_h = [], []


        # reset if not already reset (if recreated earlier)
        if 'obs_batch' not in locals():
            obs_batch, _ = envs.reset()

        for step in range(BATCH_STEPS):
            obs_t = torch.tensor(obs_batch, dtype=torch.float32).to(device)
            global_state = obs_t.view(NUM_ENVS, -1)

            with torch.no_grad():
                # seekers stacking (two agents combined into one batch of size 2*NUM_ENVS)
                s_obs = torch.cat([obs_t[:,0], obs_t[:,1]], dim=0)    # shape (2*NUM_ENVS, obs_dim)
                s_move_logits, s_yaw_logits = seeker_actor(s_obs)     # logits for both heads
                s_move_dist = Categorical(logits=s_move_logits)
                s_yaw_dist  = Categorical(logits=s_yaw_logits)

                s_move_actions = s_move_dist.sample()   # shape (2*NUM_ENVS,)
                s_yaw_actions  = s_yaw_dist.sample()    # shape (2*NUM_ENVS,)

                s_lp_move = s_move_dist.log_prob(s_move_actions)
                s_lp_yaw  = s_yaw_dist.log_prob(s_yaw_actions)
                s_lp_total = s_lp_move + s_lp_yaw        # sum used for PPO ratio (shape 2*NUM_ENVS,)

                # seeker values from centralized critic (duplicate global_state for two agents)
                glob_dup = torch.cat([global_state, global_state], dim=0)
                s_agent_ids = torch.cat([torch.zeros(NUM_ENVS, dtype=torch.long), torch.ones(NUM_ENVS, dtype=torch.long)]).to(device)
                s_val = seeker_critic(glob_dup, s_agent_ids).squeeze()   # shape (2*NUM_ENVS,)

                # Hider: single agent per env
                h_obs = obs_t[:,2]
                h_move_logits, h_yaw_logits = hider_actor(h_obs)
                h_move_dist = Categorical(logits=h_move_logits)
                h_yaw_dist  = Categorical(logits=h_yaw_logits)

                h_move_actions = h_move_dist.sample()   # shape (NUM_ENVS,)
                h_yaw_actions  = h_yaw_dist.sample()

                h_lp_move = h_move_dist.log_prob(h_move_actions)
                h_lp_yaw  = h_yaw_dist.log_prob(h_yaw_actions)
                h_lp_total = h_lp_move + h_lp_yaw

                h_agent_ids = torch.full((NUM_ENVS,), 2, dtype=torch.long).to(device)
                h_val = hider_critic(global_state, h_agent_ids).squeeze()   # shape (NUM_ENVS,)

            # Step envs
            # break stacked seeker actions into per-env s1 and s2
            s1_move, s2_move = s_move_actions.chunk(2)
            s1_yaw,  s2_yaw  = s_yaw_actions.chunk(2)

            # assemble np actions shape (NUM_ENVS, 6): [s1_move, s1_yaw, s2_move, s2_yaw, h_move, h_yaw]
            actions = np.stack([
                s1_move.cpu().numpy(), s1_yaw.cpu().numpy(),
                s2_move.cpu().numpy(), s2_yaw.cpu().numpy(),
                h_move_actions.cpu().numpy(), h_yaw_actions.cpu().numpy()
            ], axis=1).astype(np.int64)

            next_obs, rew, _, _, _ = envs.step(actions)

            # create torch tensors of next observations
            s_next = torch.tensor(next_obs, dtype=torch.float32).to(device)

            # seekers intrinsic reward (shared ICM instance but batched separately per agent)
            # note: s1_move and s1_yaw are per-env tensors (NUM_ENVS,)
            icm_rew_s1 = icm_seekers.compute_intrinsic_reward(obs_t[:,0], s_next[:,0], s1_move, s1_yaw).detach()
            icm_rew_s2 = icm_seekers.compute_intrinsic_reward(obs_t[:,1], s_next[:,1], s2_move, s2_yaw).detach()

            # hider icm: pass move + yaw actions for hider
            icm_rew_h = icm_hider.compute_intrinsic_reward(obs_t[:,2], s_next[:,2], h_move_actions, h_yaw_actions).detach()

            # normalize per ICM
            seekers_intr_all = torch.cat([icm_rew_s1, icm_rew_s2], dim=0).cpu()
            normed_seek_all = icm_norm_seekers.update_and_normalize(seekers_intr_all)
            n = icm_rew_s1.shape[0]
            norm_s1 = torch.tensor(normed_seek_all[:n], device=device, dtype=torch.float32)
            norm_s2 = torch.tensor(normed_seek_all[n:2*n], device=device, dtype=torch.float32)

            h_intr_all = icm_rew_h.cpu()
            norm_h = torch.tensor(icm_norm_hider.update_and_normalize(h_intr_all), device=device, dtype=torch.float32)

            # Store seekers: need to store move & yaw actions separately and the combined old logprob
            b_s_obs.append(s_obs)                              # (2*NUM_ENVS, obs_dim)
            b_s_act_move.append(s_move_actions)                # (2*NUM_ENVS,)
            b_s_act_yaw.append(s_yaw_actions)                  # (2*NUM_ENVS,)
            b_s_lp.append(s_lp_total.detach())                 # (2*NUM_ENVS,)
            b_s_val.append(s_val)                              # (2*NUM_ENVS,)
            s_ext = torch.tensor(rew[:,0], device=device)     # external seeker reward per env (NUM_ENVS,)
            # repeat s_ext twice and add normalized intrinsic per agent (we already compute norm_s1/norm_s2 below)
            b_s_rew.append(torch.cat([s_ext + norm_s1 * icm_scale, s_ext + norm_s2 * icm_scale], dim=0))

            # Store hider
            b_h_obs.append(h_obs)
            b_h_act_move.append(h_move_actions)
            b_h_act_yaw.append(h_yaw_actions)
            b_h_lp.append(h_lp_total.detach())
            b_h_val.append(h_val)
            b_h_rew.append(torch.tensor(rew[:,1], device=device) + norm_h * icm_scale_h)

            b_glob_s.append(torch.cat([global_state, global_state], dim=0))
            b_glob_h.append(global_state)

            obs_batch = next_obs

            # Train ICMs periodically
            if step % 32 == 0:
                # seekers' ICM loss (pass move+yaw)
                loss_s1, _ = icm_seekers.compute_loss(obs_t[:,0], s_next[:,0], s1_move, s1_yaw)
                loss_s2, _ = icm_seekers.compute_loss(obs_t[:,1], s_next[:,1], s2_move, s2_yaw)
                icm_seek_loss = 0.5 * (loss_s1 + loss_s2)
                opt_icm_s.zero_grad(); icm_seek_loss.backward(); opt_icm_s.step()

                # hider ICM loss
                loss_h, _ = icm_hider.compute_loss(obs_t[:,2], s_next[:,2], h_move_actions, h_yaw_actions)
                opt_icm_h.zero_grad(); loss_h.backward(); opt_icm_h.step()

        # After rollout: compute next values and returns / adv
        next_t = torch.tensor(next_obs, dtype=torch.float32).to(device)
        next_glob = next_t.view(NUM_ENVS, -1)

        with torch.no_grad():
            glob_dup_next = torch.cat([next_glob, next_glob], dim=0)
            n_v_s = seeker_critic(glob_dup_next, torch.cat([torch.zeros(NUM_ENVS, dtype=torch.long), torch.ones(NUM_ENVS, dtype=torch.long)]).to(device)).squeeze()

        s_ret, s_adv = compute_gae(b_s_rew, b_s_val, n_v_s)
        s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-8)

        # Flatten seekers buffers
        s_states = torch.stack(b_s_obs).reshape(-1, obs_dim)          # shape (T*2*NUM_ENVS, obs_dim)
        s_actions_move = torch.stack(b_s_act_move).reshape(-1)       # (T*2*NUM_ENVS,)
        s_actions_yaw = torch.stack(b_s_act_yaw).reshape(-1)
        s_old_lps = torch.stack(b_s_lp).reshape(-1)
        s_returns = s_ret.reshape(-1)
        s_adv_flat = s_adv.reshape(-1)
        s_globs = torch.stack(b_glob_s).reshape(-1, global_obs_dim)

        # build agent ids repeated for each rollout step (same as before)
        agent_block = torch.cat([torch.zeros(NUM_ENVS, dtype=torch.long), torch.ones(NUM_ENVS, dtype=torch.long)]).to(device)
        s_agent_ids_full = agent_block.repeat(BATCH_STEPS)[:s_states.size(0)]

        ppo_update(seeker_actor, seeker_critic, opt_s,
                s_states, s_actions_move, s_actions_yaw, s_old_lps, s_returns, s_adv_flat, s_globs,
                s_agent_ids_full, epochs=10, minibatch_size=4096)

        # Hider update
        with torch.no_grad():
            n_v_h = hider_critic(next_glob, torch.full((NUM_ENVS,), 2, dtype=torch.long).to(device)).squeeze()
        h_ret, h_adv = compute_gae(b_h_rew, b_h_val, n_v_h)
        h_adv = (h_adv - h_adv.mean()) / (h_adv.std() + 1e-8)

        h_states = torch.stack(b_h_obs).reshape(-1, obs_dim)
        h_actions_move = torch.stack(b_h_act_move).reshape(-1)
        h_actions_yaw  = torch.stack(b_h_act_yaw).reshape(-1)
        h_old_lps = torch.stack(b_h_lp).reshape(-1)
        h_returns = h_ret.reshape(-1)
        h_adv_flat = h_adv.reshape(-1)
        h_globs = torch.stack(b_glob_h).reshape(-1, global_obs_dim)
        h_agent_ids_full = torch.full((h_states.size(0),), 2, dtype=torch.long).to(device)

        ppo_update(hider_actor, hider_critic, opt_h,
                   h_states, h_actions_move, h_actions_yaw, h_old_lps, h_returns, h_adv_flat, h_globs,
                   h_agent_ids_full, epochs=10, minibatch_size=4096)

        # Logging
        dt = time.time() - start_time
        fps = (BATCH_STEPS * NUM_ENVS) / dt
        if episode % 5 == 0:
            avg_s = torch.stack(b_s_rew).mean().item() * BATCH_STEPS
            avg_h = torch.stack(b_h_rew).mean().item() * BATCH_STEPS
            print(f"✅ Ep {episode} | Lvl {current_level} | FPS: {fps:.0f} | S: {avg_s:.3f} | H: {avg_h:.3f}")

    # cleanup
    try:
        envs.close()
    except Exception:
        pass
    eval_env.close()