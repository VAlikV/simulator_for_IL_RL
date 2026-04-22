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

model = StateClassifier().to("cuda")
model.load_state_dict(torch.load("simulator_for_il_rl/models/state_model.pt", map_location="cuda"))  # или "cuda"
model.eval()

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
start_pos[1] = -0.65
start_pos[2] = 0.37

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

t = time.time()

counter = 0

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

    # print("POS:", obs["state"]["joint_pos"])
    # print("POS:", obs["state"]["ee_pos"])
    # print("QUAT:",obs["state"]["ee_quat"]) 
    # print("LIN_VEL:",obs["state"]["ee_lin_vel"])
    # print("ANG_VEL:",obs["state"]["ee_ang_vel"])
    # print("JOINTS:",obs["state"]["joint_pos"])
    # print("OBJECTS:")
    # for k in obs["objects"].keys():
    #     print(k, obs["objects"][k]["pos"])
    # print()

    image = obs["images"]["cam_state"]

    buttons = ps4_joystick.get_button_states()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("STATE", image)
    cv2.waitKey(1)

    img = Image.fromarray(image, mode="RGB")
    input_tensor = transform(img).unsqueeze(0).to("cuda")  # добавляем batch dimension: (1, 3, 224, 224)

    with torch.no_grad():
        prediction, conf, prob = model.predict(input_tensor)  # prediction shape: (1, 4)
        print("Stage: ", prediction.cpu().numpy()[0])
        print("Conf: ", conf.cpu().numpy()[0])
        print()

    if buttons['success']:
        image = cv2.resize(image, (224,224))
        cv2.imwrite(f"photos/{counter}.jpg", image)
        print(f"{counter}.jpg saved")
        counter += 1

    if buttons['terminate_episode']:
        print("Reset")

        obs, info = env.reset()
        ps4_joystick.reset()

env.close()
