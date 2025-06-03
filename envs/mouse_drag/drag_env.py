import random
import pygame
import numpy as np
from envs.base_mouse_env import BaseEnv
from gymnasium import spaces
from envs.tools import id_to_action, cosine_similarity


class DragAndDropEnv(BaseEnv):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.max_distance_o_t = None
        self.success_drop = 0
        self.target_pos = None
        self.object_pos = None
        self.dragging = False
        self.drag_offset = [0, 0]
        self.prev_cursor_pos = [0, 0]

        # Configs
        self.hp = self.max_hp
        self.object_radius = config.get("object_size", 20)
        self.target_zone_radius = config.get("target_zone_radius", 40)
        self.total_targets = config.get("total_targets", 100)

        # Reward design
        self.reward_hit = config.get("reward_hit", 0)
        self.reward_miss = config.get("reward_miss", 0)
        self.reward_success = config.get("reward_success", 100.0)
        self.reward_fail = config.get("reward_fail", 0)

        self.bg_w, self.bg_h = 299, 223
        self.icon_w, self.icon_h = 46, 64

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
                low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([self.width, self.height,  # cursor
                                    self.width, self.height,  # object
                                    self.width, self.height,  # target
                                    1.0],                     # is_holding
                                    dtype=np.float32),
                    dtype=np.float32
            )

    def _get_obs(self):
        return np.array([
            *self.cursor,
            *self.object_pos,
            *self.target_pos,
            float(self.dragging)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.step_count = 0
        self.episode_end = False
        self.success_drop = 0
        self.dragging = False
        self.drag_offset = [0, 0]
        self.generate_objects()
        self.prev_cursor_pos = [0, 0]
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

        self.max_distance_o_t = np.linalg.norm(np.array(self.object_pos) - np.array(self.target_pos))

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
        # update the cursor
        if not self.play_mode:
            self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.width)
            self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.height)

        info = {}

        object_rect = self.object_img.get_rect(topleft=self.object_pos)
        #collide_dist = np.linalg.norm(np.array(self.cursor) - np.array(self.object_pos))
        if press == 1 and object_rect.collidepoint(*self.cursor):#collide_dist <= self.object_radius:
            self.dragging = True

        if press == 0:
            self.dragging = False

        if self.dragging:
            if self.play_mode:
                dx = self.cursor[0] - self.prev_cursor_pos[0]
                dy = self.cursor[1] - self.prev_cursor_pos[1]
            self.object_pos[0] += dx
            self.object_pos[1] += dy

        reward, done, distance_reward_o_t, distance_reward_c_o = self.calculate_reward(dx, dy, press, info)

        if done:
            self.episode_end = True
        info = {"success": self.success_drop,
                "steps": self.step_count,
                "distance_reward_o_t": distance_reward_o_t,
                "distance_reward_c_o": distance_reward_c_o,
                "episode_end": self.episode_end,}
        if not self.render_mode:
            self.render()
        return self._get_obs(), reward, done, False, info

    def calculate_reward(self, dx, dy, press, info):
        reward = 0
        done = False
        dist_c_o = np.linalg.norm(np.array(self.object_pos) - np.array(self.cursor))
        dist_o_t = np.linalg.norm(np.array(self.object_pos) - np.array(self.target_pos))
        max_distance = np.linalg.norm(np.array([self.width, self.height]))
        normalized_dist_o_t = dist_o_t / self.max_distance_o_t
        normalized_dist_c_o = dist_c_o / max_distance
        distance_reward_o_t = (1 - normalized_dist_o_t)
        distance_reward_c_o = (1 - normalized_dist_c_o) * 0.1
        d_c_o = np.array([self.object_pos[0] - self.cursor[0], self.object_pos[1] - self.cursor[1]])
        d_o_t = np.array([self.target_pos[0] - self.object_pos[0], self.target_pos[1] - self.object_pos[1]])
        d_move = np.array([dx, dy])
        cosine_sim_c_o = cosine_similarity(d_c_o, d_move)
        cosine_sim_o_t = cosine_similarity(d_o_t, d_move)
        a = 0.8
        beta = 0.2

        object_rect = self.object_img.get_rect(topleft=self.object_pos)
        target_rect = self.finder_img.get_rect(topleft=self.target_pos)
        if target_rect.contains(object_rect) and press == 0 and not self.dragging:
            self.success_drop += 1
            self.generate_objects()
        reward += (a * cosine_sim_o_t + beta * distance_reward_o_t) * int(self.dragging)
        reward += (a * cosine_sim_c_o + beta * distance_reward_c_o) * (1 - int(self.dragging))

        if self.hp <= 0 or self.success_drop == self.total_targets or self.step_count >= self.max_steps:
            done = True

        if done:
            if self.success_drop == 0:
                reward += self.reward_fail
                info["result"] = "Game Over"
            else:
                reward += self.reward_success * (self.success_drop / self.total_targets)
                info["result"] = "Complete"
        return reward, done, distance_reward_o_t, distance_reward_c_o

    def render(self, mode="human"):
        surface = super().render()
        # Draw target zone
        #pygame.draw.circle(surface, (0, 255, 0), self.target_pos, self.target_zone_radius)
        if self.object_img and self.finder_img:
            obj_rect = self.object_img.get_rect(topleft=self.object_pos)
            target_rect = self.finder_img.get_rect(topleft=self.target_pos)
            surface.blit(self.finder_img, target_rect)
            surface.blit(self.object_img, obj_rect)

        # Draw object
        #pygame.draw.circle(surface, (255, 100, 100), self.object_pos, self.object_radius)

        # Draw cursor
        pygame.draw.circle(surface, (0, 0, 0), self.cursor, 5)
        if self.mode == "test":
            print(f"HP: {self.hp}", f"Score: {self.success_drop}/{self.total_targets}")

        if self.render_mode == "human":
            pygame.display.flip()
            #self.clock.tick(60)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None