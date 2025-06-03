import numpy as np
import pygame
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import gymnasium as gym
from envs.mouse_click.click_env import ClickEnv
from envs.mouse_drag.drag_env import DragAndDropEnv
from envs.tools import id_to_action

# env = ClickEnv(
#     config={
#         "width": 640,
#         "height": 480,
#         "action_space_mode": "simple",
#         "render_mode": "human",
#         "rgb": True,
#         "obs_compress": False,
#         "mode": "test",
#         "max_hp": 10000000,
#         "total_targets": 100,
#         "max_step": 10000000,
#         "obs_mode": "simple"})

env = DragAndDropEnv(
    config={
        "width": 640,
        "height": 480,
        "action_space_mode": "simple",
        "render_mode": "human",
        "rgb": True,
        "obs_compress": False,
        "mode": "test",
        "max_hp": 10000000,
        "total_targets": 100,
        "max_step": 10000000,
        "obs_mode": "simple"})
obs, _ = env.reset()
env.render()
done = False

#model = PPO.load("saved_agents/simple_action_obs_no1/ppo_click_env_20250524_204404.zip")
model = PPO.load("saved_agents/MouseDrag-v0/no1/ppo_click_env_20250603_200340.zip")

epsilon = 0.1
while not done:
    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action, _ = model.predict(obs, deterministic=True)
    #action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print("action id: ", action, "action: ", id_to_action(env.action_space_mode, action), "reward: ", reward)#, "cossim reward: ", info['cossim_reward_step'], "dist reward: ", info['dst_reward_step'])

    # plt.imshow(obs)
    # plt.title("Observation")
    # plt.axis("off")
    # plt.show()
    if done:
        print(f"Game Over! Clicked targets: {info.get('clicked_targets')}")
        obs, _ = env.reset()

env.close()