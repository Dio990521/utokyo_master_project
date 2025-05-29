import random
import pygame
import numpy as np
from envs.base_mouse_env import BaseEnv
from gymnasium import spaces
from envs.tools import id_to_action


class DragAndDropEnv(BaseEnv):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.success_drop = 0
        self.target_pos = None
        self.object_pos = None
        self.holding = False
        self.drag_offset = [0, 0]

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

    def _get_obs(self):
        return np.array([
            *self.cursor,
            *self.object_pos,
            *self.target_pos
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hp = self.max_hp
        self.step_count = 0
        self.episode_end = False
        self.success_drop = 0
        self.holding = False
        self.drag_offset = [0, 0]
        self.generate_objects()
        return self._get_obs(), {}

    def generate_objects(self):
        while True:
            self.object_pos = [
                random.randint(self.object_radius, self.width - self.object_radius),
                random.randint(self.object_radius, self.height - self.object_radius)
            ]

            self.target_pos = [
                random.randint(self.target_zone_radius, self.width - self.target_zone_radius),
                random.randint(self.target_zone_radius, self.height - self.target_zone_radius)
            ]

            dist = np.linalg.norm(np.array(self.object_pos) - np.array(self.target_pos))

            if dist > self.object_radius + self.target_zone_radius:
                break

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

        collide_dist = np.linalg.norm(np.array(self.cursor) - np.array(self.object_pos))
        if press == 1 and collide_dist <= self.object_radius:
            self.holding = True
            self.drag_offset[0] = self.cursor[0] - self.object_pos[0]
            self.drag_offset[1] = self.cursor[1] - self.object_pos[1]

        if press == 0:
            self.holding = False

        if self.holding:
            self.object_pos[0] = self.cursor[0] - self.drag_offset[0]
            self.object_pos[1] = self.cursor[1] - self.drag_offset[1]

        reward, done = self.calculate_reward(press, info)

        if done:
            self.episode_end = True

        if not self.render_mode:
            self.render()
        return self._get_obs(), reward, done, False, info

    def calculate_reward(self, press, info):
        reward = 0
        done = False
        dist = np.linalg.norm(np.array(self.object_pos) - np.array(self.target_pos))
        max_distance = np.linalg.norm(np.array([self.width, self.height]))
        normalized_dist = dist / max_distance
        distance_reward = (1 - normalized_dist) * 0.1
        if dist < self.target_zone_radius and press == 0 and not self.holding:
            self.success_drop += 1
            self.generate_objects()

        elif self.holding:
            reward += distance_reward

        if self.hp <= 0 or self.success_drop == self.total_targets or self.step_count >= self.max_steps:
            done = True

        if done:
            if self.success_drop == 0:
                reward += self.reward_fail
                info["result"] = "Game Over"
            else:
                reward += self.reward_success * (self.success_drop / self.total_targets)
                info["result"] = "Complete"
        return reward, done

    def render(self, mode="human"):
        surface = super().render()

        # Draw target zone
        pygame.draw.circle(surface, (0, 255, 0), self.target_pos, self.target_zone_radius)

        # Draw object
        pygame.draw.circle(surface, (255, 100, 100), self.object_pos, self.object_radius)

        # Draw cursor
        pygame.draw.circle(surface, (0, 0, 0), self.cursor, 5)
        if self.mode == "test":
            print(f"HP: {self.hp}", f"Score: {self.success_drop}/{self.total_targets}")

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None