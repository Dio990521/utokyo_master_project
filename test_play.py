import numpy as np
import pygame

from envs.mouse_click.click_env import ClickEnv

env = ClickEnv(config={
    "render": True,
    "hp": 100,
    "reward_hit": 1.0,
    "total_targets": 10,
    "reward_miss": -0.1,
    "reward_success": 10.0,
    "reward_fail": -1.0
})

obs, _ = env.reset()
env.render()
prev_mouse_pos = pygame.mouse.get_pos()
pressing = 0
running = True

while running:
    env.render()

    current_mouse_pos = pygame.mouse.get_pos()
    env.cursor = current_mouse_pos
    #dx = np.clip(current_mouse_pos[0] - prev_mouse_pos[0], -1, 1)
    #dy = np.clip(current_mouse_pos[1] - prev_mouse_pos[1], -1, 1)
    #dz = 0
    prev_mouse_pos = current_mouse_pos

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pressing = 1
        elif event.type == pygame.MOUSEBUTTONUP:
            pressing = 0

    action = np.array([0,0,0,pressing])#np.array([dx, dy, dz, pressing], dtype=np.int32)

    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action} -> Reward: {reward}, HP: {env.hp}, Score: {env.score}")

    if done:
        print(f"Game Over: {info.get('result')}")
        obs, _ = env.reset()
        prev_mouse_pos = pygame.mouse.get_pos()

env.close()