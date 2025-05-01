import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import cv2

#import pyautogui

from envs.action_tool import id_to_action
from envs.base_mouse_env import BaseEnv

winPos = 100
os.environ["SDL_VIDEO_WINDOW_POS"]= "{},{}".format(winPos, winPos)

class ClickEnv(BaseEnv):

    def __init__(self, config=None):
        super().__init__(config=config)

        # Configs3
        self.target_radius = config.get("target_radius", 20)
        self.total_targets = config.get("total_targets", 100)

        # Reward design
        self.reward_hit = config.get("reward_hit", 1.0)
        self.reward_miss = config.get("reward_miss", -0.1)
        self.reward_success = config.get("reward_success", 10.0)
        self.reward_fail = config.get("reward_fail", -1.0)

        self.hp = self.max_hp
        self.score = 0
        self.target_pos = self._random_position()

    def _random_position(self):
        return [
            random.randint(self.target_radius, self.width - self.target_radius),
            random.randint(self.target_radius, self.height - self.target_radius)
        ]

    def _get_obs(self):
        if not self.render_mode :
            if not self.surface:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            obs = pygame.surfarray.array3d(self.surface)
        else:
            if not self.window:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            obs = pygame.surfarray.array3d(self.window)
        obs = np.transpose(obs, (1, 0, 2))  #(w, h, c)->(h, w, c)
        if not self.rgb:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        #obs = obs / 255.0
        return obs.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.score = 0
        self.target_pos = self._random_position()
        return self._get_obs(), {}

    def step(self, action):
        if not self.play_mode:
            dx, dy, dz, press = id_to_action(action)
        else:
            dx, dy, dz, press = action

        # update the cursor
        if not self.play_mode:
            self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.width - 1)
            self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.height - 1)

        dist = np.linalg.norm(np.array(self.target_pos) - np.array(self.cursor))
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
        max_distance = np.linalg.norm(np.array([self.width, self.height]))
        normalized_dist = dist / max_distance
        reward += (1 - normalized_dist) * 0.1
        return self._get_obs(), reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("ClickEnv")
                #pyautogui.moveTo(winPos + self.width / 2, winPos + self.height / 2)
                pygame.mouse.set_pos((self.width // 2, self.height // 2))

            self.window.fill((255, 255, 255))  # white background

            # Draw target
            pygame.draw.circle(self.window, (255, 100, 100), self.target_pos, self.target_radius)
            # Draw cursor
            pygame.draw.circle(self.window, (0, 0, 0), self.cursor, 5)

            # Draw status text
            font = pygame.font.SysFont("Arial", 24)
            hp_text = font.render(f"HP: {self.hp}", True, (0, 0, 0))
            score_text = font.render(f"Score: {self.score}/{self.total_targets}", True, (0, 0, 0))
            self.window.blit(hp_text, (10, 10))
            self.window.blit(score_text, (10, 30))

            pygame.display.flip()
        else:
            if self.surface is None:
                pygame.init()
                self.surface = pygame.Surface((self.width, self.height))
            self.surface.fill((255, 255, 255))  # white background

            # Draw target
            pygame.draw.circle(self.surface, (255, 100, 100), self.target_pos, self.target_radius)
            # Draw cursor
            pygame.draw.circle(self.surface, (0, 0, 0), self.cursor, 5)

            # Draw status text
            font = pygame.font.SysFont("Arial", 24)
            hp_text = font.render(f"HP: {self.hp}", True, (0, 0, 0))
            score_text = font.render(f"Score: {self.score}/{self.total_targets}", True, (0, 0, 0))
            self.surface.blit(hp_text, (10, 10))
            self.surface.blit(score_text, (10, 30))


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
