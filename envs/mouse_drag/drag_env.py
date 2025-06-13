import random
import pygame
import numpy as np
from envs.base_mouse_env import BaseEnv
from gymnasium import spaces
from envs.tools import cosine_similarity


class DragAndDropEnv(BaseEnv):

    def __init__(self, config=None):
        super().__init__(config)

        # Task-specific attributes
        self.prev_cursor_pos = [0, 0]
        self.object_pos = None
        self.target_pos = None
        self.hp = self.max_hp
        self.dragging = False
        self.total_targets = config.get("total_targets", 100)

        # Reward design
        self.reward_success = config.get("reward_success", 100.0)

        if self.obs_mode == "simple":
            # [d_cursor_obj_x, d_cursor_obj_y, d_obj_target_x, d_obj_target_y, is_dragging]
            self.observation_space = spaces.Box(
                low=np.array([-self.width, -self.height, -self.width, -self.height, 0], dtype=np.float32),
                high=np.array([self.width, self.height, self.width, self.height, 1.0], dtype=np.float32),
                dtype=np.float32
            )

    def _get_obs(self):
        if self.obs_mode == "simple":
            # d_cursor_object = ((np.array(self.object_pos) - np.array(self.cursor)) /
            #                    np.array([self.width, self.height]))
            # d_object_target = ((np.array(self.target_pos) - np.array(self.object_pos)) /
            #                    np.array([self.width, self.height]))
            # return np.array([
            #     *d_cursor_object,
            #     *d_object_target,
            #     float(self.dragging)
            # ], dtype=np.float32)
            cursor_norm = np.array(self.cursor) / np.array([self.width, self.height])
            object_norm = np.array(self.object_pos) / np.array([self.width, self.height])
            target_norm = np.array(self.target_pos) / np.array([self.width, self.height])

            return np.array([
                *cursor_norm,
                *object_norm,
                *target_norm,
                float(self.dragging)
            ], dtype=np.float32)
        return self._get_image_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.success = 0
        self.dragging = False
        self.generate_objects()
        self.prev_cursor_pos = [0, 0]
        if not self.render_mode: self.render()
        return self._get_obs(), {}

    def generate_objects(self):
        bg_x = random.randint(0, self.width - self.bg_w)
        bg_y = random.randint(0, self.height - self.bg_h)
        self.target_pos = [bg_x, bg_y]
        bg_rect = pygame.Rect(bg_x, bg_y, self.bg_w, self.bg_h)

        while True:
            icon_x = random.randint(0, self.width - self.icon_w)
            icon_y = random.randint(0, self.height - self.icon_h)
            icon_rect = pygame.Rect(icon_x, icon_y, self.icon_w, self.icon_h)
            if not icon_rect.colliderect(bg_rect):
                self.object_pos = [icon_x, icon_y]
                break

    def step(self, action):
        self.step_count += 1
        dx, dy, _, press = self._decode_action(action)
        self._update_cursor(dx, dy)

        # Handle dragging state
        object_rect = pygame.Rect(self.object_pos[0], self.object_pos[1], self.icon_w, self.icon_h)
        if press == 1 and object_rect.collidepoint(*self.cursor):
            self.dragging = True

        if press == 0:
            self.dragging = False

        if self.dragging:
            if self.play_mode:
                dx = self.cursor[0] - self.prev_cursor_pos[0]
                dy = self.cursor[1] - self.prev_cursor_pos[1]
            self.object_pos[0] += dx
            self.object_pos[1] += dy

        reward, done = self._calculate_reward(dx, dy, press)
        if done:
            self.episode_end = True
            self.episode_count += 1

        info = {"success": self.success, "steps": self.step_count, "episode_end": self.episode_end}
        if not self.render_mode: self.render()
        return self._get_obs(), reward, done, False, info

    def _calculate_reward(self, dx, dy, press):
        reward = 0
        done = False

        # Check for successful drop
        object_rect = self.object_img.get_rect(topleft=self.object_pos)
        target_rect = self.finder_img.get_rect(topleft=self.target_pos)
        if not self.dragging and press == 0 and target_rect.contains(object_rect):
            self.success += 1
            self.generate_objects()

        # Reward shaping based on phase (approaching vs. dragging)
        d_move = np.array([dx, dy])
        alpha, beta = 0.2, 0.8
        if self.dragging:
            # Phase 2: Drag object to target
            center_obj = np.array(object_rect.center)
            center_target = np.array(target_rect.center)
            d_obj_target = center_target - center_obj
            dist_o_t = np.linalg.norm(d_obj_target)

            max_dist_o_t = np.linalg.norm([self.width, self.height])
            reward += (1 - dist_o_t / max_dist_o_t) * alpha  # Proximity reward
            reward += cosine_similarity(d_obj_target, d_move) * beta  # Direction reward
        else:
            # Phase 1: Approach the object
            center_obj = np.array(object_rect.center)
            d_cursor_obj = center_obj - np.array(self.cursor)
            dist_c_o = np.linalg.norm(d_cursor_obj)

            max_dist_c_o = np.linalg.norm([self.width, self.height])
            reward += (1 - dist_c_o / max_dist_c_o) * alpha  # Proximity reward
            reward += cosine_similarity(d_cursor_obj, d_move) * beta  # Direction reward

        # Episode termination
        if self.hp <= 0 or self.success >= self.total_targets or self.step_count >= self.max_steps:
            done = True
            if self.success > 0:
                reward += self.reward_success * (self.success / self.total_targets)

        return reward, done

    def render(self, mode="human"):
        surface = super().render()
        if self.object_pos and self.target_pos:
            surface.blit(self.finder_img, self.finder_img.get_rect(topleft=self.target_pos))
            surface.blit(self.object_img, self.object_img.get_rect(topleft=self.object_pos))
        super().draw_cursor(surface)

        if self.mode == "test-show-ui":
            font = pygame.font.SysFont("Arial", 24)
            surface.blit(font.render(f"HP: {self.hp}", True, (0, 0, 0)), (10, 10))
            surface.blit(font.render(f"Score: {self.success}/{self.total_targets}", True, (0, 0, 0)), (10, 30))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])