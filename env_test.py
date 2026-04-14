import numpy as np
from env import UR10Env
import matplotlib.pyplot as plt
import cv2
import time

env = UR10Env(xml_path="robot/scene.xml",
            sim_timestep = 0.001,
            control_hz = 30.0,
            mode = "realtime",   # "realtime" | "fast"
            max_episode_steps = 1000,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

t = time.time()

plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for _ in range(1001):
    action = np.array([1.57, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)

    imgs = obs["images"]

    for ax, (name, img) in zip(axes, imgs.items()):
        ax.clear()
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    plt.pause(0.001)

    if terminated or truncated:
        print("Episode ended:", terminated, truncated, info)
        obs, info = env.reset()

        print("Время:", time.time() - t)

env.close()

plt.ioff()
plt.show()