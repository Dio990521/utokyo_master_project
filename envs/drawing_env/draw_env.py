import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import os
import random
import pygame

from envs.drawing_env.tools.image_process import find_starting_point, calculate_pixel_similarity, \
    calculate_block_reward, visualize_obs, calculate_iou_similarity


def _decode_action(action):
    """
    Decodes a discrete action (0-18) into dx, dy, pen_state, and stop_action.
    - Actions 0-8: Pen is UP.
    - Actions 9-17: Pen is DOWN.
    - Action 18: STOP.

    The 9 movements correspond to a 3x3 grid around the cursor.
    """
    #if action == 18:
    #    return 0, 0, 0, 1 # dx, dy, is_pen_down, is_stop

    is_pen_down = action >= 9

    sub_action = action % 9
    dx = (sub_action % 3) - 1  # Results in -1, 0, or 1
    dy = (sub_action // 3) - 1  # Results in -1, 0, or 1

    return dx, dy, int(is_pen_down), 0


class DrawingAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, config=None):
        super(DrawingAgentEnv, self).__init__()
        self.episode_end = False
        self.last_pixel_similarity = None
        self.canvas_size = config.get("canvas_size", [100, 100])
        self.max_steps = config.get("max_steps", 1000)
        self.render_mode = config.get("render_mode", "human")
        self.current_step = 0
        self.max_hp = config.get("max_hp", 1000000)
        self.stroke_budget = config.get("stroke_budget", 1)
        self.used_budgets = 0
        self.similarity_weight = config.get("similarity_weight", 1)
        self.budget_weight = config.get("budget_weight", 1)

        self.hp = self.max_hp
        self.target_sketches_path = config.get("target_sketches_path", None)
        self.target_sketches = self._load_target_sketches()
        if not self.target_sketches:
            raise ValueError(f"No target sketches found in {self.target_sketches_path}")

        self.block_reward_levels = [16, 8, 4]
        self.current_block_level_index = 0
        self.current_block_size = self.block_reward_levels[self.current_block_level_index]
        self.level_up_thresholds = {
            16: 0.5,
            8: 0.75,
        }

        self.action_space = spaces.Discrete(18) # 0-17 for movement, 18 for stop
        self.observation_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(4, *self.canvas_size), # (4, H, W)
            dtype=np.float32
        )
        # self.observation_space = spaces.Dict({
        #     "image": spaces.Box(low=0, high=255, shape=(3, *self.canvas_size), dtype=np.uint8),
        #     # canvas, target, pen_mask
        #     "vector": spaces.Box(low=0, high=self.stroke_budget, shape=(1,), dtype=np.float32)  # budget
        # })

        self.canvas = None
        self.target_sketch = None
        self.cursor = [0, 0]
        self.is_pen_down = False
        self.pen_was_down = False

        self.window = None
        self.clock = None
        self.canvas_surface = None
        self.target_sketch_surface = None

    def _load_target_sketches(self):
        sketches = []
        if not os.path.exists(self.target_sketches_path):
            print(f"Warning: Target sketches path '{self.target_sketches_path}' does not exist.")
            return sketches
        for filename in os.listdir(self.target_sketches_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(self.target_sketches_path, filename)
                try:
                    sketch = Image.open(filepath).resize(self.canvas_size).convert('L')
                    #255 (white) -> 1.0, 0 (black) -> 0.0
                    sketch_array = np.array(sketch)
                    normalized_sketch = (sketch_array / 255.0).astype(np.float32)
                    sketches.append(normalized_sketch)
                except Exception as e:
                    print(f"Error loading sketch {filepath}: {e}")
        print(f"Loaded {len(sketches)} target sketches.")
        return sketches

    def _update_block_level(self, final_similarity):
        if self.current_block_level_index < len(self.block_reward_levels) - 1:
            threshold = self.level_up_thresholds.get(self.current_block_size)

            if threshold is not None and final_similarity > threshold:
                self.current_block_level_index += 1
                self.current_block_size = self.block_reward_levels[self.current_block_level_index]
                print(f"Grid is updated to: {self.current_block_size}x{self.current_block_size}")

    def _get_obs(self):
        pen_position_mask = np.full(self.canvas_size, 0.0, dtype=np.float32)
        y, x = self.cursor[1], self.cursor[0]
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            pen_position_mask[y, x] = 1.0

        normalized_budget = self.stroke_budget / 255.0
        stroke_budget_channel = np.full(self.canvas_size, normalized_budget, dtype=np.float32)

        observation = np.stack([
            self.canvas.copy(),
            self.target_sketch.copy(),
            pen_position_mask,
            stroke_budget_channel
        ], axis=-1)
        observation = observation.transpose(2, 0, 1)
        return observation
        # image_obs = np.stack([
        #     self.canvas.copy(),
        #     self.target_sketch.copy(),
        #     pen_position_mask
        # ], axis=-1).transpose(2, 0, 1)
        #
        # vector_obs = np.array([self.stroke_budget-1], dtype=np.float32)
        #
        # return {"image": image_obs, "vector": vector_obs}

    def _get_info(self):
        return {
            "similarity": self.last_pixel_similarity,
            "episode_end": self.episode_end,
            "used_budgets": self.used_budgets,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.hp = self.max_hp
        self.canvas = np.full(self.canvas_size, 1.0, dtype=np.float32)
        self.episode_end = False
        self.used_budgets = 0
        self.pen_was_down = False
        self.is_pen_down = False

        self.target_sketch = random.choice(self.target_sketches)
        self.cursor = find_starting_point(self.target_sketch)
        self.last_pixel_similarity = 0
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()

                self.window = pygame.display.set_mode(
                    (self.canvas_size[0] * 2 + 10, self.canvas_size[1]))
                pygame.display.set_caption("Drawing Sketch RL")
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.canvas_surface = pygame.surfarray.make_surface(
                np.stack([self.canvas] * 3, axis=-1))
            self.target_sketch_surface = pygame.surfarray.make_surface(np.stack([self.target_sketch] * 3, axis=-1))

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1
        dx, dy, is_pen_down, is_stop_action = _decode_action(action)
        is_pen_down = bool(is_pen_down)
        is_stop_action = bool(is_stop_action)

        reward = 0.0
        terminated = False
        truncated = False

        if is_stop_action:
            terminated = True
            reward += calculate_block_reward(self.canvas, self.target_sketch, self.current_block_size)
        else:
            self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.canvas_size[0] - 1)
            self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.canvas_size[1] - 1)

            if self.pen_was_down and not is_pen_down:
                self.used_budgets += 1
                self.used_budgets = min(255, self.used_budgets)

            self.is_pen_down = is_pen_down

            # if self.pen_lift_budget < 0:
            #     self.is_pen_down = False
            #     terminated = True

            self.pen_was_down = self.is_pen_down

            if self.is_pen_down:
                x, y = self.cursor
                if int(self.canvas[y, x]) == 1:
                    self.canvas[y, x] = 0.0
                    reward += 0.1 if int(self.target_sketch[y, x]) == 0 else -0.1
                    if self.target_sketch[y, x] != 0: self.hp -=1

        current_pixel_similarity = calculate_iou_similarity(self.canvas, self.target_sketch)
        #reward += self.similarity_weight * (current_pixel_similarity - self.last_pixel_similarity) * 10.0
        self.last_pixel_similarity = current_pixel_similarity

        if self.current_step >= self.max_steps or self.hp <= 0:
            truncated = True

        if terminated or truncated:
            self.episode_end = True
            #if self.used_budgets <= self.stroke_budget:
            #    reward += 1.0 * self.last_pixel_similarity

            if not is_stop_action:
                #reward += self.similarity_weight * self.last_pixel_similarity
                reward += calculate_block_reward(self.canvas, self.target_sketch, self.current_block_size)
            self._update_block_level(self.last_pixel_similarity)

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode: self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.canvas_size[0] * 2 + 10, self.canvas_size[1]))
                pygame.display.set_caption("Drawing Agent RL")
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise Exception("Pygame window closed by user.")

            pil_canvas_rgb = Image.fromarray(self.canvas, 'L').convert('RGB')
            self.canvas_surface = pygame.image.fromstring(pil_canvas_rgb.tobytes(), pil_canvas_rgb.size,
                                                          pil_canvas_rgb.mode)

            pil_target_rgb = Image.fromarray(self.target_sketch, 'L').convert('RGB')
            self.target_sketch_surface = pygame.image.fromstring(pil_target_rgb.tobytes(), pil_target_rgb.size,
                                                                 pil_target_rgb.mode)

            self.window.fill((105, 105, 105))

            self.window.blit(self.target_sketch_surface, (0, 0))

            self.window.blit(self.canvas_surface, (self.canvas_size[0] + 10, 0))

            cursor_color = (255, 0, 0) if self.is_pen_down else (0, 0, 255)
            cursor_radius = 2
            pygame.draw.circle(self.window, cursor_color,
                               (self.canvas_size[0] + 10 + self.cursor[0], self.cursor[1]),
                               cursor_radius)

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            img_rgb = Image.fromarray(self.canvas, 'L').convert('RGB')
            draw = ImageDraw.Draw(img_rgb)
            cursor_color = (255, 0, 0) if self.is_pen_down else (0, 0, 255)
            cursor_radius = 5
            draw.ellipse((self.cursor[0] - cursor_radius, self.cursor[1] - cursor_radius,
                          self.cursor[0] + cursor_radius, self.cursor[1] + cursor_radius),
                         fill=cursor_color, outline=cursor_color)
            return np.array(img_rgb)

        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.canvas_surface = None
            self.target_sketch_surface = None