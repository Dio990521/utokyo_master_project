import numpy as np
import pygame
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import gymnasium as gym
from envs.mouse_click.click_env import ClickEnv

env = ClickEnv(
    config={
    "render_mode": "human",
})

print(env.render_mode)
obs, _ = env.reset()
env.render()
done = False
model = PPO.load("ppo_click_env_1.zip")

while not done:
    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, truncated, info = env.step(action)
    # plt.imshow(obs)
    # plt.title("Observation")
    # plt.axis("off")
    # plt.show()
    if done:
        print(f"Game Over: {info.get('result')}")
        obs, _ = env.reset()

env.close()