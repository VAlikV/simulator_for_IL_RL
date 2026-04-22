import numpy as np
from simulator_for_il_rl.env import AssemblingEnv
import matplotlib.pyplot as plt
import cv2
import time
from rc10_api.ps4_joystick import PS4Joystick

env = AssemblingEnv(xml_path="scene.xml",
            sim_timestep = 0.001,
            control_hz = 20.0,
            mode = "realtime",   # "realtime" | "fast"
            max_episode_steps = -1,
            use_task_space=True,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

start_pos = np.concatenate([obs["state"]["ee_pos"],obs["state"]["ee_quat"], [0]])
start_pos[0] = 0.1
start_pos[1] = -0.5
start_pos[2] = 0.35

ps4_joystick = PS4Joystick(
    max_speed=0.05,
    max_rot_speed=0.5,
    deadzone=0.05,
    alpha=0.3,
    poll_rate=100,
    x_init=0.1,
    y_init=-0.5,
    z_init=0.35,
    roll_init=np.pi,
    pitch_init=0.0,
    yaw_init=np.pi/2
)

t = time.time()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for _ in range(10001):
    # s = np.sin(_/(2*np.pi))/100
    # start_pos[2] -= s

    x, y, z, roll, pitch, yaw = ps4_joystick.get_joystick()

    gripper = ps4_joystick.get_gripper_state()

    start_pos[0] = x
    start_pos[1] = y
    start_pos[2] = z

    if gripper == -1:
        start_pos[-1] = 0
    else:
        start_pos[-1] = gripper
    # start_pos[5] = yaw

    obs, reward, terminated, truncated, info = env.step(start_pos)

    imgs = obs["images"]

    # for ax, (name, img) in zip(axes, imgs.items()):
    #     ax.clear()
    #     ax.imshow(img)
    #     ax.set_title(name)
    #     ax.axis("off")

    # plt.pause(0.001)

    print("POS:", obs["state"]["joint_pos"])
    # print("POS:", obs["state"]["ee_pos"])
    # print("QUAT:",obs["state"]["ee_quat"]) 
    # print("LIN_VEL:",obs["state"]["ee_lin_vel"])
    # print("ANG_VEL:",obs["state"]["ee_ang_vel"])
    # print("JOINTS:",obs["state"]["joint_pos"])
    # print("OBJECTS:")
    # for k in obs["objects"].keys():
    #     print(k, obs["objects"][k]["pos"])
    # print()

    if terminated or truncated:
        print("Episode ended:", terminated, truncated, info)
        obs, info = env.reset()

        print("Время:", time.time() - t)

    # if _ % 100 == 0:
    #     obs, info = env.reset()

env.close()

# plt.ioff()
# plt.show()