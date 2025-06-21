import numpy as np
import pygame

from envs.mouse_click.click_env import ClickEnv
from envs.mouse_drag.drag_env import DragAndDropEnv
from envs.mouse_dropdown.dropdown_env import DropdownEnv

env = None
ENV = "click" # click; drag; dropdown
if ENV == "click":
    env = ClickEnv(
        config={
        "render_mode": "human",
        "play_mode": True
    })
elif ENV == "drag":
    env = DragAndDropEnv(
        config={
            "render_mode": "human",
            "play_mode": True,
            "max_hp": 10000000,
            "total_targets": 100,
        }
    )
elif ENV == "dropdown":
    env = DropdownEnv(
        config={
            "render_mode": "human",
            "play_mode": True,
            "max_hp": 10000000,
            "total_targets": 100,
            "obs_mode": "simple"
        }
    )

obs, _ = env.reset()
env.render()
pressing = 0
running = True
env.prev_cursor_pos = pygame.mouse.get_pos()
while running:
    env.render()
    current_mouse_pos = pygame.mouse.get_pos()
    env.cursor = list(current_mouse_pos)

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
    #print(obs)
    print(f"Action: {action} -> Reward: {reward}, HP: {env.hp}, Score: {env.success}")
    env.prev_cursor_pos = current_mouse_pos
    if done:
        print(f"Game Over: {info.get('result')}")
        obs, _ = env.reset()

env.close()