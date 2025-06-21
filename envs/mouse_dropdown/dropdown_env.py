import random
import pygame
import numpy as np
from gymnasium import spaces

from envs.base_mouse_env import BaseEnv


class DropdownEnv(BaseEnv):

    def __init__(self, config=None):
        super().__init__(config)
        if config is None: config = {}

        # Menu properties
        menu_w = 120
        menu_h = 30
        menu_x = (self.width - menu_w) // 2
        menu_y = (self.height - menu_h) // 3
        self.menu_rect = pygame.Rect(menu_x, menu_y, menu_w, menu_h)
        self.options = ["A", "B", "C"]
        self.goal_option = "C"#random.choice(self.options)
        self.option_rects = [
            pygame.Rect(self.menu_rect.x, self.menu_rect.y + (i + 1) * self.menu_rect.h, self.menu_rect.w,
                        self.menu_rect.h)
            for i in range(len(self.options))
        ]

        # Task-specific state
        self.hp = self.max_hp
        self.selected_option = None
        self.menu_open = False
        self.success = 0
        self.first_open = False

        # Reward
        self.reward_success = config.get("reward_success", 100.0)

        # Font for rendering
        self.font = pygame.font.SysFont(None, 24)

        # This environment has a unique observation structure, so it's defined here
        if self.obs_mode == "simple":
            # [cursor_x, cursor_y, menu_open_flag, selected_option_idx]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, len(self.options) - 1], dtype=np.float32),
                dtype=np.float32
            )

    def _get_obs(self):
        cursor_x, cursor_y = self.cursor
        menu_open_val = float(self.menu_open)

        menu_cx = self.menu_rect.centerx / self.width
        menu_cy = self.menu_rect.centery / self.height

        obs_list = [
            cursor_x / self.width,
            cursor_y / self.height,
            menu_cx,
            menu_cy,
        ]

        if self.menu_open:
            for rect in self.option_rects:
                opt_cx = rect.centerx / self.width
                opt_cy = rect.centery / self.height
                obs_list.extend([opt_cx, opt_cy])
        else:
           obs_list.extend([-1] * len(self.option_rects) * 2)

        goal_index = self.options.index(self.goal_option)
        obs_list.append(goal_index)
        obs_list.append(menu_open_val)

        return np.array(obs_list, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.selected_option = None
        self.menu_open = False
        self.success = 0
        self.episode_end = False
        self.first_open = False
        self.cursor = [self.width // 2, self.height // 2]
        #self.cursor = [self.menu_rect.centerx, self.menu_rect.centery]
        #random.shuffle(self.options)
        #self.goal_option = random.choice(self.options)
        if not self.render_mode: self.render()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        dx, dy, _, press = self._decode_action(action)
        self._update_cursor(dx, dy)

        reward = 0
        done = False

        if press == 1:
            # Click on menu header to open/close
            if self.menu_rect.collidepoint(self.cursor):
                if not self.menu_open and not self.first_open:
                    reward += 10.0
                    self.first_open = True
                if self.menu_open:
                    reward -= 1.0
                self.menu_open = not self.menu_open
            # If menu is open, check for clicks on options
            elif self.menu_open:
                for i, option_rect in enumerate(self.option_rects):
                    if option_rect.collidepoint(self.cursor):
                        self.selected_option = self.options[i]
                        self.menu_open = False
                        if self.selected_option == self.goal_option:
                            reward += self.reward_success
                            self.success += 1
                        else:
                            reward -= 1.0
                            self.hp -= 1
                        break

        # Episode termination
        if self.success > 0 or self.hp <= 0 or self.step_count >= self.max_steps:
            done = True

        self.episode_end = done
        info = {"success": self.success, "steps": self.step_count, "episode_end": self.episode_end}

        if not self.render_mode: self.render()
        return self._get_obs(), reward, done, False, info

    # def _calculate_reward(self, dx, dy, press):
    #     if not self.menu_open:
    #         # Phase 1: Approach the object
    #         center_obj = np.array(object_rect.center)
    #         d_cursor_obj = center_obj - np.array(self.cursor)
    #         dist_c_o = np.linalg.norm(d_cursor_obj)
    #
    #         max_dist_c_o = np.linalg.norm([self.width, self.height])
    #         reward += (1 - dist_c_o / max_dist_c_o) * alpha  # Proximity reward
    #         reward += cosine_similarity(d_cursor_obj, d_move) * beta  # Direction reward

    def render(self):
        surface = super().render()

        # Draw menu header
        pygame.draw.rect(surface, (100, 100, 255), self.menu_rect)
        text = self.selected_option if self.selected_option else "Select Option"
        text_surf = self.font.render(text, True, (255, 255, 255))
        surface.blit(text_surf, (self.menu_rect.x + 5, self.menu_rect.y + 5))

        # Draw options if menu is open
        if self.menu_open:
            for i, option in enumerate(self.options):
                pygame.draw.rect(surface, (200, 200, 200), self.option_rects[i])
                option_text = self.font.render(option, True, (0, 0, 0))
                surface.blit(option_text, (self.option_rects[i].x + 5, self.option_rects[i].y + 5))
        super().draw_cursor(surface)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return surface