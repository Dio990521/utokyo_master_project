import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None):
        super().__init__()
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self.max_hp = config.get("hp", 100)
        self.play_mode = config.get("play_mode", False)
        self.render_mode = config.get("render_mode", None)
        self.max_steps = config.get("max_step", 1000)
        self.window = None
        self.surface = None
        self.mode = config.get("mode", "play")
        self.cursor = [self.width // 2, self.height // 2]
        self.rgb = config.get("rgb", True)

        # action space: [dx, dy, dz, press/release]
        # dx/dy/dz: [-1, 0, 1]
        # press: 1; release: 0
        self.action_space = spaces.Discrete(54)
        if not self.rgb:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8
            )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError