import numpy as np
import pygame

from envs.mouse_click.click_env import ClickEnv

env = ClickEnv(
    config={
    "render_mode": "human",
    "play_mode": True
})

obs, _ = env.reset()
env.render()
pressing = 0
running = True

while running:
    env.render()

    current_mouse_pos = pygame.mouse.get_pos()
    env.cursor = current_mouse_pos

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pressing = 1
        elif event.type == pygame.MOUSEBUTTONUP:
            pressing = 0

    action = np.array([0,0,0,pressing])

    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action} -> Reward: {reward}, HP: {env.hp}, Score: {env.score}")

    if done:
        print(f"Game Over: {info.get('result')}")
        obs, _ = env.reset()

env.close()