import os
import numpy as np
import pygame
import random
import cv2
from gymnasium import spaces
#import pyautogui
import matplotlib.pyplot as plt

from envs.tools import id_to_action, cosine_similarity
from envs.base_mouse_env import BaseEnv

winPos = 100
os.environ["SDL_VIDEO_WINDOW_POS"]= "{},{}".format(winPos, winPos)

class ClickEnv(BaseEnv):

    def __init__(self, config=None):
        super().__init__(config=config)

        # Configs
        self.hp = self.max_hp
        self.target_radius = config.get("target_radius", 20)
        self.total_targets = config.get("total_targets", 100)

        # Reward design
        self.reward_hit = config.get("reward_hit", 0)
        self.reward_miss = config.get("reward_miss", 0)
        self.reward_success = config.get("reward_success", 100.0)
        self.reward_fail = config.get("reward_fail", 0)

        self.clicked_targets = 0
        self.click_times = 0
        self.distance_reward = 0
        self.cossim_reward = 0
        self.target_pos = self._random_position()

        if self.obs_mode == "image":
            self.obs_channels = 3 if self.rgb else 1
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_height, self.obs_width, self.obs_channels),
                dtype=np.uint8
            )
        elif self.obs_mode == "simple":
            self.observation_space = spaces.Box(
                low=np.array([-self.width, -self.height], dtype=np.float32),
                high=np.array([self.width, self.height], dtype=np.float32),
                dtype=np.float32
            )

    def _random_position(self):
        return [
            random.randint(self.target_radius, self.width - self.target_radius),
            random.randint(self.target_radius, self.height - self.target_radius)
        ]

    def _get_obs(self):
        if self.obs_mode == "simple":
            dx = self.target_pos[0] - self.cursor[0]
            dy = self.target_pos[1] - self.cursor[1]
            return np.array([dx, dy], dtype=np.float32)

        if not self.render_mode :
            if not self.surface:
                return np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)
            obs = pygame.surfarray.array3d(self.surface)
        else:
            if not self.window:
                return np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)
            obs = pygame.surfarray.array3d(self.window)

        obs = np.transpose(obs, (1, 0, 2))  #(w, h, c)->(h, w, c)
        if self.obs_compress:
            obs = cv2.resize(obs, (self.obs_height, self.obs_width), interpolation=cv2.INTER_AREA)
        if not self.rgb:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.expand_dims(obs, axis=-1)
        # plt.imshow(obs)
        # plt.title("Observation")
        # plt.axis("off")
        # plt.show()
        return obs.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.clicked_targets = 0
        self.step_count = 0
        self.click_times = 0
        self.distance_reward = 0
        self.cossim_reward = 0
        self.target_pos = self._random_position()
        self.episode_end = False
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        dx, dy, dz, press = 0, 0, 0, 0
        if not self.play_mode:
            action = id_to_action(self.action_space_mode, action)
            if len(action) == 3:
                dx, dy, press = action
            elif len(action) == 4:
                dx, dy, dz, press = action
        else:
            dx, dy, dz, press = action

        self.click_times += press

        # update the cursor
        if not self.play_mode:
            self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.width - 1)
            self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.height - 1)

        dist = np.linalg.norm(np.array(self.target_pos) - np.array(self.cursor))
        info = {}
        reward, done, dist_reward, cossim_reward = self.calculate_reward(dx, dy, dist, press, info)
        if done:
            self.episode_end = True
        info = {"click_times": self.click_times,
                "distance_reward": self.distance_reward,
                "cossim_reward": self.cossim_reward,
                "success": self.clicked_targets,
                "steps": self.step_count,
                "dst_reward_step": dist_reward,
                "cossim_reward_step": cossim_reward,
                "episode_end": self.episode_end,}
        if not self.render_mode:
            self.render()
        return self._get_obs(), reward, done, False, info

    def calculate_reward(self, dx, dy, dist, press, info):
        reward = 0
        done = False
        if dist <= self.target_radius and press == 1:
            reward += self.reward_hit
            self.clicked_targets += 1
            self.target_pos = self._random_position()
        elif dist > self.target_radius and press == 1:
            reward += self.reward_miss
            self.hp -= 1

        if self.hp <= 0 or self.clicked_targets == self.total_targets or self.step_count >= self.max_steps:
            done = True

        if done:
            if self.clicked_targets == 0:
                reward += self.reward_fail
                info["result"] = "Game Over"
            else:
                reward += self.reward_success * (self.clicked_targets / self.total_targets)
                info["result"] = "Complete"
        max_distance = np.linalg.norm(np.array([self.width, self.height]))
        normalized_dist = dist / max_distance
        distance_reward = (1 - normalized_dist) * 0.1
        # reward += distance_reward
        # self.distance_reward += distance_reward

        d_target = np.array([self.target_pos[0] - self.cursor[0], self.target_pos[1] - self.cursor[1]])
        d_move = np.array([dx, dy])
        cosine_sim = cosine_similarity(d_target, d_move)
        a = 0.8
        beta = 0.2
        reward += a * cosine_sim + beta * distance_reward
        self.distance_reward += beta * distance_reward
        self.cossim_reward += a * cosine_sim
        return reward, done, beta * distance_reward, a * cosine_sim

    def render(self):
        surface = super().render()

        # Draw target and cursor
        pygame.draw.circle(surface, (255, 100, 100), self.target_pos, self.target_radius)
        pygame.draw.circle(surface, (0, 0, 0), self.cursor, 5)

        if self.mode == "test-show-ui":
            # Draw status text
            font = pygame.font.SysFont("Arial", 24)
            surface.blit(font.render(f"HP: {self.hp}", True, (0, 0, 0)), (10, 10))
            surface.blit(font.render(f"Score: {self.clicked_targets}/{self.total_targets}", True, (0, 0, 0)), (10, 30))
        elif self.mode == "test":
            print(f"HP: {self.hp}", f"Score: {self.clicked_targets}/{self.total_targets}")

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
