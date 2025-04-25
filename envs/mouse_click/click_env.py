import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import pyautogui

winPos = 100
os.environ["SDL_VIDEO_WINDOW_POS"]= "{},{}".format(winPos, winPos)

class ClickEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, config=None):
        # Configs
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self.target_radius = config.get("target_radius", 20)
        self.total_targets = config.get("total_targets", 100)
        self.max_hp = config.get("hp", 100)

        # Reward design
        self.reward_hit = config.get("reward_hit", 1.0)
        self.reward_miss = config.get("reward_miss", -0.1)
        self.reward_success = config.get("reward_success", 10.0)
        self.reward_fail = config.get("reward_fail", -1.0)

        self.hp = self.max_hp
        self.score = 0
        self.target_pos = self._random_position()

        self.render_mode = render_mode
        self.render_enabled = config.get("render", True)
        self.window = None
        self.mode = config.get("mode", "play")
        self.cursor = [self.width // 2, self.height // 2]

        # action space: [dx, dy, dz, press/release]
        # dx/dy/dz: [-1, 0, 1]
        # press: 1; release: 0
        self.action_space = spaces.Discrete(54)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8
        )

    def _random_position(self):
        return [
            random.randint(self.target_radius, self.width - self.target_radius),
            random.randint(self.target_radius, self.height - self.target_radius)
        ]

    def _get_obs(self):
        if self.window is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs = pygame.surfarray.array3d(self.window)
        #obs = np.transpose(obs, (1, 0, 2))  #(w, h, c)->(h, w, c)
        return obs.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.score = 0
        self.target_pos = self._random_position()
        return self._get_obs(), {}

    def step(self, action):
        dx, dy, dz, press = action.astype(int)
        # update the cursor
        #self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.width - 1)
        #self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.height - 1)

        dist = np.linalg.norm(np.array(self.target_pos) - np.array(self.cursor))
        print(self.cursor, dist, self.target_pos, self.target_radius)
        done = False
        info = {}
        reward = 0
        if dist <= self.target_radius and press == 1:
            reward = self.reward_hit
            self.score += 1
            self.target_pos = self._random_position()
        elif dist > self.target_radius and press == 1:
            reward = self.reward_miss
            self.hp -= 1

        if self.hp <= 0:
            done = True
            reward += self.reward_fail
            info["result"] = "Game Over"
        elif self.score >= self.total_targets:
            done = True
            reward += self.reward_success * (self.score / self.total_targets)
            info["result"] = "Success"

        return self._get_obs(), reward, done, False, info

    def render(self):
        if not self.render_enabled:
            return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("ClickEnv")
            pyautogui.moveTo(winPos + self.width / 2, winPos + self.height / 2)
            pygame.mouse.set_pos((self.width // 2, self.height // 2))

        self.window.fill((255, 255, 255))  # white background

        # Draw target
        pygame.draw.circle(self.window, (255, 100, 100), self.target_pos, self.target_radius)

        # Draw status text
        font = pygame.font.SysFont("Arial", 24)
        hp_text = font.render(f"HP: {self.hp}", True, (0, 0, 0))
        score_text = font.render(f"Score: {self.score}/{self.total_targets}", True, (0, 0, 0))
        self.window.blit(hp_text, (10, 10))
        self.window.blit(score_text, (10, 30))

        pygame.display.flip()

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
