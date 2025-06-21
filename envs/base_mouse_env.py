import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

from envs.tools import id_to_action


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None):
        super().__init__()
        pygame.init()
        # Screen and asset dimensions
        self.width = config.get("width", 84)
        self.height = config.get("height", 84)
        self.bg_w, self.bg_h = 299, 223
        self.icon_w, self.icon_h = 46, 64

        # Environment configuration
        self.play_mode = config.get("play_mode", False)
        self.render_mode = config.get("render_mode", None)
        self.mode = config.get("mode", "training")
        self.obs_mode = config.get("obs_mode", "image")
        self.rgb = config.get("rgb", True)
        self.obs_compress = config.get("obs_compress", False)

        # Episode and state tracking
        self.max_hp = config.get("max_hp", 100)
        self.max_steps = config.get("max_step", 1000000)
        self.step_count = 0
        self.episode_count = 0
        self.episode_end = False
        self.success = 0
        self.cursor = [self.width // 2, self.height // 2]

        # Pygame setup
        pygame.init()
        self.window = None
        self.surface = None
        self.clock = pygame.time.Clock()
        self.finder_img, self.object_img = None, None

        # Asset paths
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.file_icon_path = os.path.join(root_dir, "envs/assets", "file.png")
        self.bg_path = os.path.join(root_dir, "envs/assets", "bg.jpg")

        # Observation space
        if self.obs_compress:
            self.obs_width, self.obs_height = 120, 160
        else:
            self.obs_width, self.obs_height = self.width, self.height

        if self.obs_mode == "image":
            self.obs_channels = 3 if self.rgb else 1
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.obs_height, self.obs_width, self.obs_channels),
                dtype=np.uint8
            )

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.file_icon_path = os.path.join(root_dir, "envs/assets", "file.png")
        self.bg_path = os.path.join(root_dir, "envs/assets", "bg.jpg")

        # Action space: [dx, dy, dz, press/release]
        # dx/dy/dz: [-1, 0, 1]
        # press: 1; release: 0
        # delete [0, 0, 0, 0]
        self.action_space_mode = config.get("action_space_mode", "complex")
        if self.action_space_mode == "complex":
            self.action_space = spaces.Discrete(53)
        else:  # simple
            self.action_space = spaces.Discrete(17)

    def _decode_action(self, action):
        if self.play_mode:
            return action

        decoded = id_to_action(self.action_space_mode, action)
        if len(decoded) == 3:
            dx, dy, press = decoded
            dz = 0
        else:
            dx, dy, dz, press = decoded
        return dx, dy, dz, press

    def _update_cursor(self, dx, dy):
        if not self.play_mode:
            self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.width - 1)
            self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.height - 1)

    def _get_image_obs(self):
        source_surface = self.window if self.render_mode == "human" else self.surface
        if source_surface is None:
            return np.zeros((self.obs_height, self.obs_width, 3 if self.rgb else 1), dtype=np.uint8)

        obs = pygame.surfarray.array3d(source_surface)
        obs = np.transpose(obs, (1, 0, 2))  # (w, h, c) -> (h, w, c)

        if self.obs_compress:
            obs = cv2.resize(obs, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)

        if not self.rgb:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.expand_dims(obs, axis=-1)

        return obs.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.success = 0
        self.episode_end = False

    def step(self, action):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Mouse Environment")
                pygame.mouse.set_pos((self.width // 2, self.height // 2))
                self.object_img = pygame.image.load(self.file_icon_path).convert_alpha()
                self.finder_img = pygame.image.load(self.bg_path).convert_alpha()
            surface = self.window
        else:
            if self.surface is None:
                self.surface = pygame.Surface((self.width, self.height))
            surface = self.surface
            self.object_img = pygame.Surface((self.icon_w, self.icon_h), pygame.SRCALPHA)
            self.finder_img = pygame.Surface((self.bg_w, self.bg_h), pygame.SRCALPHA)

        surface.fill((247, 247, 247))  # white background

        return surface

    def draw_cursor(self, surface):
        pygame.draw.circle(surface, (0, 0, 0), self.cursor, 5) # Draw cursor

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None