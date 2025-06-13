import numpy as np
import pygame
import random
from gymnasium import spaces
from envs.tools import cosine_similarity
from envs.base_mouse_env import BaseEnv

class ClickEnv(BaseEnv):

    def __init__(self, config=None):
        super().__init__(config)

        # Task-specific attributes
        self.hp = self.max_hp
        self.target_radius = config.get("target_radius", 20)
        self.total_targets = config.get("total_targets", 100)
        self.click_times = 0
        self.target_pos = self._random_position()

        # Reward design
        self.reward_success = config.get("reward_success", 100.0)

        if self.obs_mode == "simple":
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
            return np.array([
                self.cursor[0] / self.width,
                self.cursor[1] / self.height,
                self.target_pos[0] / self.width,
                self.target_pos[1] / self.height
            ], dtype=np.float32)
        return self._get_image_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.click_times = 0
        self.target_pos = self._random_position()
        if not self.render_mode: self.render()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        dx, dy, _, press = self._decode_action(action)
        self._update_cursor(dx, dy)
        self.click_times += press

        dist = np.linalg.norm(np.array(self.target_pos) - np.array(self.cursor))
        reward, done = self._calculate_reward(dx, dy, dist, press)

        if done:
            self.episode_end = True

        info = {
            "click_times": self.click_times,
            "success": self.success,
            "steps": self.step_count,
            "episode_end": self.episode_end,
        }

        if not self.render_mode: self.render()
        return self._get_obs(), reward, done, False, info

    def _calculate_reward(self, dx, dy, dist, press):
        reward = 0
        done = False

        # Click outcome
        if press == 1:
            if dist <= self.target_radius:
                self.success += 1
                self.target_pos = self._random_position()
            else:
                self.hp -= 1

        # Proximity and direction shaping rewards
        d_target = np.array(self.target_pos) - np.array(self.cursor)
        d_move = np.array([dx, dy])

        max_dist = np.linalg.norm([self.width, self.height])
        distance_reward = (1 - (dist / max_dist)) * 0.2
        cosine_sim = cosine_similarity(d_target, d_move) * 0.8
        reward += distance_reward + cosine_sim

        # Episode termination
        if self.hp <= 0 or self.success >= self.total_targets or self.step_count >= self.max_steps:
            done = True
            if self.success > 0:
                reward += self.reward_success * (self.success / self.total_targets)

        return reward, done

    def render(self):
        surface = super().render()
        pygame.draw.circle(surface, (255, 100, 100), self.target_pos, self.target_radius)
        super().draw_cursor(surface)
        if self.mode == "test-show-ui":
            font = pygame.font.SysFont("Arial", 24)
            surface.blit(font.render(f"HP: {self.hp}", True, (0, 0, 0)), (10, 10))
            surface.blit(font.render(f"Score: {self.success}/{self.total_targets}", True, (0, 0, 0)), (10, 30))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
