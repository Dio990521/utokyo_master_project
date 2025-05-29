import gymnasium as gym
from gymnasium import spaces
import pygame

class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None):
        super().__init__()
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self.max_hp = config.get("max_hp", 100)
        self.play_mode = config.get("play_mode", False)
        self.render_mode = config.get("render_mode", None)
        self.max_steps = config.get("max_step", 1000)
        self.mode = config.get("mode", "training")
        self.window = None
        self.surface = None
        self.step_count = 0
        self.mode = config.get("mode", "test")
        self.cursor = [self.width // 2, self.height // 2]
        self.rgb = config.get("rgb", True)
        self.obs_compress = config.get("obs_compress", False)
        self.episode_end = False
        self.obs_mode = config.get("obs_mode", "image")
        self.clock = pygame.time.Clock()
        self.action_space_mode = config.get("action_space_mode", "complex")
        if self.obs_compress:
            self.obs_width = 120
            self.obs_height = 160
        else:
            self.obs_width = self.width
            self.obs_height = self.height

        # action space: [dx, dy, dz, press/release]
        # dx/dy/dz: [-1, 0, 1]
        # press: 1; release: 0
        # delete [0, 0, 0, 0]
        if self.action_space_mode == "complex":
            self.action_space = spaces.Discrete(53)
        elif self.action_space_mode == "simple":
            self.action_space = spaces.Discrete(17)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

    def step(self, action):
        raise NotImplementedError

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
        return surface

    def close(self):
        raise NotImplementedError