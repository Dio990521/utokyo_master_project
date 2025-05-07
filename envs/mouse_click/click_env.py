import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import cv2
from sympy.abc import alpha

#import pyautogui

from envs.tools import id_to_action, cosine_similarity
from envs.base_mouse_env import BaseEnv

winPos = 100
os.environ["SDL_VIDEO_WINDOW_POS"]= "{},{}".format(winPos, winPos)

class ClickEnv(BaseEnv):

    def __init__(self, config=None):
        super().__init__(config=config)

        # Configs
        self.step_count = 0
        self.target_radius = config.get("target_radius", 20)
        self.total_targets = config.get("total_targets", 100)

        # Reward design
        self.reward_hit = config.get("reward_hit", 1)
        self.reward_miss = config.get("reward_miss", -0.1)
        self.reward_success = config.get("reward_success", 10.0)
        self.reward_fail = config.get("reward_fail", -1)

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
        return obs.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.score = 0
        self.step_count = 0
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
        info = {}
        reward, done = self.calculate_reward(dx, dy, dist, press, info)
        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True
        #print("reward", reward, "done", done, "step", self.step_count)
        return self._get_obs(), reward, done, False, info

    def calculate_reward(self, dx, dy, dist, press, info):
        reward = 0
        done = False
        if dist <= self.target_radius and press == 1:
            reward += self.reward_hit
            self.score += 1
            self.target_pos = self._random_position()
        elif dist > self.target_radius and press == 1:
            reward += self.reward_miss
            self.hp -= 1

        if self.hp <= 0:
            done = True
            reward += self.reward_fail
            info["result"] = "Game Over"
        elif self.score >= self.total_targets:
            done = True
            reward += self.reward_success * (self.score / self.total_targets)
            info["result"] = "Success"
        # max_distance = np.linalg.norm(np.array([self.width, self.height]))
        # normalized_dist = dist / max_distance
        # distance_reward = (1 - normalized_dist) * 0.1
        #
        # d_target = np.array([self.target_pos[0] - self.cursor[0], self.target_pos[1] - self.cursor[1]])
        # d_move = np.array([dx, dy])
        # cosine_sim = cosine_similarity(d_target, d_move)
        # a = 0.8
        # beta = 0.2
        # reward += a * max(0, cosine_sim) + beta * distance_reward
        return reward, done

    def render(self):
        pygame.init()

        if self.render_mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("ClickEnv")
                pygame.mouse.set_pos((self.width // 2, self.height // 2))
            surface = self.window
        else:
            if self.surface is None:
                self.surface = pygame.Surface((self.width, self.height))
            surface = self.surface

        surface.fill((255, 255, 255))  # white background

        # Draw target and cursor
        pygame.draw.circle(surface, (255, 100, 100), self.target_pos, self.target_radius)
        pygame.draw.circle(surface, (0, 0, 0), self.cursor, 5)

        # Draw status text
        font = pygame.font.SysFont("Arial", 24)
        surface.blit(font.render(f"HP: {self.hp}", True, (0, 0, 0)), (10, 10))
        surface.blit(font.render(f"Score: {self.score}/{self.total_targets}", True, (0, 0, 0)), (10, 30))

        if self.render_mode == "human":
            pygame.display.flip()


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
