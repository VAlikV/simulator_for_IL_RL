import numpy as np
from simulator_for_il_rl.env import AssemblingEnv
import matplotlib.pyplot as plt
import cv2
import time
from rc10_api.ps4_joystick import PS4Joystick
from scipy.spatial.transform import Rotation

from simulator_for_il_rl.state_models import StateClassifier, transform
import torch
from PIL import Image

import pandas as pd

def encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes()

def save_trajectory(path, traj, success=True, index=0):
    """
    traj: list of dict
    {
        "images.cam_front": np.ndarray (H,W,3)
        "images.cam_side": np.ndarray (H,W,3)
        "images.cam_gripper": np.ndarray (H,W,3)
        "images.cam_state": np.ndarray (H,W,3)
        "state.joint_pos": np.array
        "state.joint_vel": np.array
        "state.ee_pos": np.array
        "state.ee_quat": np.array
        "state.ee_lin_vel": np.array
        "state.ee_ang_vel": np.array
    }
    """

    rows = []

    for t, step in enumerate(traj):
        row = {
            "images.cam_front": encode_image(step["images"]["cam_front"]),
            "images.cam_side": encode_image(step["images"]["cam_side"]),
            "images.cam_gripper": encode_image(step["images"]["cam_gripper"]),
            "images.cam_state": encode_image(step["images"]["cam_state"]),
            "state.joint_pos": step["state"]["joint_pos"],
            "state.joint_vel": step["state"]["joint_vel"],
            "state.ee_pos": step["state"]["ee_pos"],
            "state.ee_quat": step["state"]["ee_quat"],
            "state.ee_lin_vel": step["state"]["ee_lin_vel"],
            "state.ee_ang_vel": step["state"]["ee_ang_vel"],
        }

        # if "action" in step:
        #     row["action"] = step["action"].astype(np.float32)

        rows.append(row)

    df = pd.DataFrame(rows)

    # имя файла
    prefix = "success" if success else "fail"
    df.to_parquet(f"{path}/{prefix}_{index}.parquet", index=False)

# ====================================================================

env = AssemblingEnv(xml_path="scene.xml",
            sim_timestep = 0.001,
            control_hz = 20.0,
            mode = "realtime",   # "realtime" | "fast"
            max_episode_steps = -1,
            use_task_space=True,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

ps4_joystick = PS4Joystick(
    max_speed=0.05,
    max_rot_speed=0.5,
    deadzone=0.05,
    alpha=0.3,
    poll_rate=100,
    x_init=0.1,
    y_init=-0.65,
    z_init=0.37,
    roll_init=np.pi,
    pitch_init=0.0,
    yaw_init=np.pi/2
)

# ====================================================================

obs, info = env.reset()

start_pos = np.concatenate([obs["state"]["ee_pos"],obs["state"]["ee_quat"], [0]])
start_pos[0] = 0.1
start_pos[1] = -0.65
start_pos[2] = 0.37

counter = 0

traj = []

for _ in range(1000001):

    x, y, z, roll, pitch, yaw = ps4_joystick.get_joystick()
    gripper = ps4_joystick.get_gripper_state()

    euler = np.array([roll, pitch, yaw])
    r = Rotation.from_euler('xyz', euler, degrees=False)
    quat_xyzw = r.as_quat()

    start_pos[0] = x
    start_pos[1] = y
    start_pos[2] = z

    start_pos[3] = quat_xyzw[3]
    start_pos[4] = quat_xyzw[0]
    start_pos[5] = quat_xyzw[1]
    start_pos[6] = quat_xyzw[2]

    if gripper == -1:
        start_pos[-1] = 0
    else:
        start_pos[-1] = gripper

    obs, reward, terminated, truncated, info = env.step(start_pos)

    traj.append(obs)

    image = obs["images"]["cam_state"]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("STATE", image)
    cv2.waitKey(1)

    buttons = ps4_joystick.get_button_states()

    if buttons['success']: # X
        print("Save ", counter)

        save_trajectory("outputs", traj, success=False, index=counter)

        obs, info = env.reset()
        ps4_joystick.reset()

        traj = []
        counter += 1

    if buttons['terminate_episode']: # B
        print("Reset")

        obs, info = env.reset()
        ps4_joystick.reset()

        traj = []

env.close()
