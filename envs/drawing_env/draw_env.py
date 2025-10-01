import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import os
import random
import pygame

from envs.drawing_env.tools.image_process import find_starting_point, calculate_pixel_similarity, \
    calculate_block_reward, visualize_obs, calculate_iou_similarity, \
    calculate_qualified_block_similarity, calculate_density_cap_reward


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
        self.stroke_budget = config.get("stroke_budget", 1)
        self.used_budgets = 0
        self.similarity_weight = config.get("similarity_weight", 1)
        self.budget_weight = config.get("budget_weight", 1)
        self.block_similarity = 0
        self.block_reward = 0
        self.use_step_similarity_reward = config.get("use_step_similarity_reward", False)
        self.use_stroke_reward = config.get("use_stroke_reward", False)
        self.block_reward_scale = config.get("block_reward_scale", 1.0)
        self.stroke_reward_scale = config.get("stroke_reward_scale", 1.0)
        self.stroke_penalty = config.get("stroke_penalty", -20.0)
        self.step_rewards = 0
        self.r_stroke_hyper = config.get("r_stroke_hyper", 100)
        self.target_sketches_path = config.get("target_sketches_path", None)
        self.target_sketches = self._load_target_sketches()
        if not self.target_sketches:
            raise ValueError(f"No target sketches found in {self.target_sketches_path}")

        self.block_size = config.get("block_size", 16)
        self.local_reward_block_size = config.get("local_reward_block_size", 1)
        self.use_local_reward_block = config.get("use_local_reward_block", False)
        # self.block_reward_levels = [16, 8, 4, 2]
        # self.current_block_level_index = 0
        # self.current_block_size = self.block_reward_levels[self.current_block_level_index]
        # self.level_up_thresholds = {
        #     16: 0.9,
        #     8: 0.8,
        #     4: 0.8,
        # }

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
        self.delta_similarity_history = []
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

    def _update_block_level(self, block_score):
        if self.current_block_level_index < len(self.block_reward_levels) - 1:
            threshold = self.level_up_thresholds.get(self.current_block_size)

            if threshold is not None and block_score > threshold:
                self.current_block_level_index += 1
                self.current_block_size = self.block_reward_levels[self.current_block_level_index]
                print(
                    f"--- Level UP! Grid difficulty increased to: {self.current_block_size}x{self.current_block_size} ---")

    def _get_obs(self):
        pen_position_mask = np.full(self.canvas_size, 0.0, dtype=np.float32)
        y, x = self.cursor[1], self.cursor[0]
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            pen_position_mask[y, x] = 1.0

        normalized_budget = (self.stroke_budget-1) / 255.0
        stroke_budget_channel = np.full(self.canvas_size, normalized_budget, dtype=np.float32)

        observation = np.stack([
            self.canvas.copy(),
            self.target_sketch.copy(),
            pen_position_mask,
            stroke_budget_channel
        ], axis=-1)
        observation = observation.transpose(2, 0, 1)
        return observation

    def _get_info(self):
        info_dict = {
            "similarity": self.last_pixel_similarity,
            "episode_end": self.episode_end,
            "used_budgets": self.used_budgets,
            "block_similarity": self.block_similarity,
            "block_reward": self.block_reward,
            "step_rewards": self.step_rewards,
        }
        if self.episode_end:
            info_dict["delta_similarity_history"] = self.delta_similarity_history
        return info_dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.delta_similarity_history = []
        self.step_rewards = 0
        self.current_step = 0
        self.canvas = np.full(self.canvas_size, 1.0, dtype=np.float32)
        self.episode_end = False
        self.used_budgets = 0
        self.pen_was_down = False
        self.is_pen_down = False
        self.block_similarity = 0
        self.block_reward = 0

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

            canvas_to_render = (self.canvas * 255).astype(np.uint8)
            target_to_render = (self.target_sketch * 255).astype(np.uint8)

            self.canvas_surface = pygame.surfarray.make_surface(
                np.stack([canvas_to_render] * 3, axis=-1))
            self.target_sketch_surface = pygame.surfarray.make_surface(
                np.stack([target_to_render] * 3, axis=-1))
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _calculate_reward_scale(self, block_size: int) -> float:
        max_size = self.block_reward_levels[0]
        base_scale = 0.5
        exponent = 2.0

        if block_size <= 0:
            block_size = 1

        scale = base_scale * (max_size / block_size) ** exponent

        return scale

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
            if not self.use_local_reward_block:
                if self.is_pen_down:
                    x, y = self.cursor
                    if int(self.canvas[y, x]) == 1:
                        self.canvas[y, x] = 0.0
                        if not self.use_step_similarity_reward:
                            if int(self.target_sketch[y, x]) == 0:
                                reward += 0.1
                                self.step_rewards += 0.1
                            else:
                                reward -= 0.1
                                self.step_rewards -= 0.1
            else:
                tactical_reward = 0.0
                if self.is_pen_down:
                    x, y = self.cursor

                    if np.isclose(self.canvas[y, x], 1.0):
                        self.canvas[y, x] = 0.0

                        tactical_reward = calculate_density_cap_reward(
                            self.canvas,
                            self.target_sketch,
                            self.cursor,
                            self.local_reward_block_size
                        )
                        self.step_rewards += tactical_reward
                reward += tactical_reward

        current_pixel_similarity = calculate_iou_similarity(self.canvas, self.target_sketch)
        delta = current_pixel_similarity - self.last_pixel_similarity
        if self.use_step_similarity_reward:
            reward += self.similarity_weight * delta * 100.0
            self.step_rewards += self.similarity_weight * delta * 100.0
        self.delta_similarity_history.append(delta)
        self.last_pixel_similarity = current_pixel_similarity

        if self.current_step >= self.max_steps:
            truncated = True

        if terminated or truncated:
            self.episode_end = True
            if self.use_stroke_reward:
                # if self.used_budgets <= self.stroke_budget:
                #     reward += self.stroke_reward_scale * self.last_pixel_similarity
                # else:
                #     reward += self.stroke_penalty
                if self.used_budgets > 0:
                    reward += self.r_stroke_hyper / self.used_budgets * current_pixel_similarity

            self.block_similarity = calculate_block_reward(self.canvas, self.target_sketch, self.block_size)
            self.block_reward = self.block_similarity * self.block_reward_scale
            reward += self.block_reward
            #self._update_block_level(self.block_similarity)

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

            canvas_to_render = (self.canvas * 255).astype(np.uint8)
            target_to_render = (self.target_sketch * 255).astype(np.uint8)
            pil_canvas_rgb = Image.fromarray(canvas_to_render, 'L').convert('RGB')
            self.canvas_surface = pygame.image.fromstring(pil_canvas_rgb.tobytes(), pil_canvas_rgb.size,
                                                          pil_canvas_rgb.mode)

            pil_target_rgb = Image.fromarray(target_to_render, 'L').convert('RGB')
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