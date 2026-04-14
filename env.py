import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import time


class UR10Env(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"]}

    joints_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "left_driver_joint" #, "right_driver_joint"
    ]

    actuators_names = [
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3", "fingers_actuator"
    ]

    gripper_actuator_name = "fingers_actuator"
    gripper_joint_name = "left_driver_joint"

    camera_names = ["cam_front", "cam_side"]

    objects_names = ["bottom", "mid", "cap"]

    initial_pose = [1.57, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0]

    def __init__(
                self,
                xml_path: str,
                sim_timestep: float = 0.001,
                control_hz: float = 10.0,
                mode: str = "realtime",   # "realtime" | "fast"
                max_episode_steps: int = 1000,
                use_task_space: bool = False,
                render_mode=None,           # "human" | "rgb_array" | "all"
    ):
        super().__init__()

        assert mode in ["realtime", "fast"]
        self.mode = mode
        self.realtime = (mode == "realtime")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ===================== timing =====================
        self._setup_idx()
        self._setup_spaces()

        # ===================== timing =====================
        self.model.opt.timestep = sim_timestep
        self.sim_timestep = self.model.opt.timestep

        self.control_hz = control_hz
        self.control_dt = 1.0 / self.control_hz

        # согласование control_hz и physics
        self.frame_skip = max(1, int(round(self.control_dt / self.sim_timestep)))
        self.sim_dt = self.frame_skip * self.sim_timestep

        self.next_step_time = None

        # ===================== rendering =====================
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        if self.render_mode == "all" or self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)

        # ===================== misc =====================
        self.use_task_space = use_task_space
        self.step_count = 0
        self.max_episode_steps = max_episode_steps

    # ======================================================================

    def _setup_idx(self):
        self.joints_qpos_idx = []
        self.joints_qvel_idx = []
        for name in self.joints_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]

            self.joints_qpos_idx.append(qpos_adr)
            self.joints_qvel_idx.append(qvel_adr)

        self.joints_qpos_idx = np.array(self.joints_qpos_idx)
        self.joints_qvel_idx = np.array(self.joints_qvel_idx)

        # ----------------------

        self.actuator_idx = []
        for name in self.actuators_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.actuator_idx.append(actuator_id)

        self.actuator_idx = np.array(self.actuator_idx)

        # ----------------------

        self.objects_idx = []
        for name in self.objects_names:
            object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.objects_idx.append(object_id)

        self.objects_idx = np.array(self.objects_idx)

    # ======================================================================

    def _setup_spaces(self):
        self.action_dim = self.actuator_idx.shape[0]

        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # ----------------------
        n_joints = len(self.joints_qpos_idx)

        state_space = spaces.Dict({
            "joint_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_joints,), dtype=np.float32
            ),
            "joint_vel": spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_joints,), dtype=np.float32
            ),
        })

        # ----------------------
        objects_space = {}

        for obj in self.objects_names:
            objects_space[obj] = spaces.Dict({
                f"{obj}_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                f"{obj}_vel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
            })

        objects_space = spaces.Dict(objects_space)

        # ----------------------
        image_space = {}

        if (self.render_mode == "rgb_array") or (self.render_mode == "all"):
            # лучше взять из renderer
            H = self.renderer.height
            W = self.renderer.width

            for cam in self.camera_names:
                image_space[cam] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(H, W, 3),
                    dtype=np.uint8
                )

        image_space = spaces.Dict(image_space)

        # ----------------------
        self.observation_space = spaces.Dict({
            "state": state_space,
            "objects": objects_space,
            "images": image_space,
        })

    # ======================================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        for i, id in enumerate(self.joints_qpos_idx):
            self.data.qpos[id] = self.initial_pose[i]

        mujoco.mj_forward(self.model, self.data)

        if ((self.render_mode == "human") or (self.render_mode == "all")) and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.step_count = 0

        if self.realtime:
            self.next_step_time = time.perf_counter() + self.control_dt
        else:
            self.next_step_time = None

        return self._get_obs(), {}

    # ======================================================================

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[:] = action

        # physics step
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # viewer update
        if ((self.render_mode == "human") or (self.render_mode == "all")) and self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()

        self.step_count += 1

        obs = self._get_obs()
        
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        # ===================== realtime pacing =====================
        if self.realtime:
            now = time.perf_counter()
            lag = now - self.next_step_time

            if lag < 0.0:
                time.sleep(-lag)
                self.next_step_time += self.control_dt
            else:
                if lag > self.control_dt:
                    print(f"Accumulated lag: {lag:.3f}s")

                if lag > 5.0 * self.control_dt:
                    self.next_step_time = now + self.control_dt
                else:
                    self.next_step_time += self.control_dt

        return obs, reward, terminated, truncated, {}

    # ======================================================================

    def _apply_action(self, action):
        pass

    # ======================================================================

    def _get_obs(self):

        obs = {"state":{}, "images": {}, "objects":{}}

        obs["state"]["joint_pos"] = self.data.qpos[self.joints_qpos_idx]
        obs["state"]["joint_vel"] = self.data.qvel[self.joints_qvel_idx]

        for i, obj in enumerate(self.objects_names):
            obs["objects"][f"{obj}_pos"] = self.data.qpos[self.objects_idx[i]:self.objects_idx[i]+7]
            obs["objects"][f"{obj}_vel"] = self.data.qvel[self.objects_idx[i]:self.objects_idx[i]+6]

        images = self.render()
        if images is not None:
            for cam, img in images.items():
                obs["images"][cam] = img

        return obs

    # ======================================================================

    def render_cameras(self):
        if self.renderer is None:
            raise RuntimeError("render_cameras requires render_mode='rgb_array'.")

        images = {}
        for cam_name in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam_name)
            images[cam_name] = self.renderer.render().copy()
        return images

    # ======================================================================

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_cameras()

        elif self.render_mode == "all":
            return self.render_cameras()

        return None

    # ======================================================================

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None