import random
import pygame
import numpy as np
from gymnasium import spaces

from envs.base_mouse_env import BaseEnv


class DropdownEnv(BaseEnv):
    """An environment where the agent must open a dropdown menu and select the correct option."""

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
        self.goal_option = "B"
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

        # Reward
        self.reward_success = config.get("reward_success", 1.0)

        # Font for rendering
        self.font = pygame.font.SysFont(None, 24)

        # This environment has a unique observation structure, so it's defined here
        if self.obs_mode == "simple":
            # [cursor_x, cursor_y, menu_open_flag, selected_option_idx]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, -1], dtype=np.float32),
                high=np.array([self.width, self.height, 1, len(self.options) - 1], dtype=np.float32),
                dtype=np.float32
            )

    def _get_obs(self):
        if self.obs_mode == "simple":
            menu_open_val = float(self.menu_open)

            dx_menu = (self.menu_rect.centerx - self.cursor[0]) / self.width
            dy_menu = (self.menu_rect.centery - self.cursor[1]) / self.height

            obs_list = [menu_open_val, dx_menu, dy_menu]

            if self.menu_open:
                for option_rect in self.option_rects:
                    dx_opt = (option_rect.centerx - self.cursor[0]) / self.width
                    dy_opt = (option_rect.centery - self.cursor[1]) / self.height
                    obs_list.extend([dx_opt, dy_opt])
            else:
                obs_list.extend([0.0] * len(self.option_rects) * 2)

            return np.array(obs_list, dtype=np.float32)

        return self._get_image_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.selected_option = None
        self.menu_open = False
        self.success = 0
        self.episode_end = False
        self.cursor = [self.width // 2, self.height // 2]
        random.shuffle(self.options)
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
                self.menu_open = not self.menu_open
            # If menu is open, check for clicks on options
            elif self.menu_open:
                for i, option_rect in enumerate(self.option_rects):
                    if option_rect.collidepoint(self.cursor):
                        self.selected_option = self.options[i]
                        self.menu_open = False
                        if self.selected_option == self.goal_option:
                            reward = self.reward_success
                            self.success += 1
                        else:
                            reward = -0.1  # Penalty for wrong selection
                            self.hp -= 1
                        break

        # Episode termination
        if self.success > 0 or self.hp <= 0 or self.step_count >= self.max_steps:
            done = True

        self.episode_end = done
        info = {"success": self.success, "steps": self.step_count, "episode_end": self.episode_end}

        if not self.render_mode: self.render()
        return self._get_obs(), reward, done, False, info

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