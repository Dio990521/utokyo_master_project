import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import os
import random
import pygame

from envs.drawing_env.tools.image_process import find_starting_point, calculate_pixel_similarity, \
    calculate_block_reward, visualize_obs, calculate_iou_similarity, \
    calculate_qualified_block_similarity, calculate_density_cap_reward, calculate_reward_map


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
        self._init_config(config)
        self._init_state_variables()
        self.render_scale = 10

        self.action_space = spaces.Discrete(18) # 0-17 for movement, 18 for stop
        num_obs_channels = 4 if self.use_budget_channel else 3
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(num_obs_channels, *self.canvas_size), dtype=np.float32
        )

        self.window = None
        self.clock = None

    def _init_config(self, config):
        self.canvas_size = config.get("canvas_size", [32, 32])
        self.max_steps = config.get("max_steps", 1000)
        self.render_mode = config.get("render_mode", None)
        self.stroke_budget = config.get("stroke_budget", 1)
        self.dynamic_budget_channel = config.get("dynamic_budget_channel", False)
        self.use_budget_channel = config.get("use_budget_channel", True)

        self.brush_size = config.get("brush_size", 1)
        self.target_square_size = config.get("target_square_size", 15)
        self.use_reward_map_reward = config.get("use_reward_map_reward", False)
        self.reward_map_on_target = config.get("reward_map_on_target", 0.1)
        self.reward_map_near_target = config.get("reward_map_near_target", 0.0)
        self.reward_map_far_target = config.get("reward_map_far_target", -0.1)
        self.reward_map_near_distance = config.get("reward_map_near_distance", 2)

        self.use_step_similarity_reward = config.get("use_step_similarity_reward", False)
        self.use_stroke_reward = config.get("use_stroke_reward", False)
        self.block_reward_scale = config.get("block_reward_scale", 1.0)
        self.block_size = config.get("block_size", 16)
        self.stroke_reward_scale = config.get("stroke_reward_scale", 1.0)
        self.r_stroke_hyper = config.get("r_stroke_hyper", 100)
        self.similarity_weight = config.get("similarity_weight", 100.0)

        self.target_sketches_path = config.get("target_sketches_path", None)
        self.specific_sketch_file = config.get("specific_sketch_file", None)
        # self.target_sketches = self._load_target_sketches()
        # if not self.target_sketches:
        #     raise ValueError("No target sketches were loaded!")

    def _init_state_variables(self):
        self.current_step = 0
        self.canvas = np.full(self.canvas_size, 1.0, dtype=np.float32)
        self.used_budgets = 0
        self.is_pen_down = False
        self.pen_was_down = False
        self.episode_end = False
        self.last_pixel_similarity = 0
        self.block_similarity = 0
        self.block_reward = 0
        self.step_rewards = 0
        self.delta_similarity_history = []
        self.target_sketch = None
        self.cursor = [0, 0]

    def _load_target_sketches(self):
        sketches = []
        if self.specific_sketch_file:
            if os.path.exists(self.specific_sketch_file):
                sketches.append(self._load_sketch_from_path(self.specific_sketch_file))
        elif self.target_sketches_path and os.path.exists(self.target_sketches_path):
            for filename in os.listdir(self.target_sketches_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(self.target_sketches_path, filename)
                    sketches.append(self._load_sketch_from_path(filepath))
        return sketches

    def _load_sketch_from_path(self, filepath):
        sketch = Image.open(filepath).resize(self.canvas_size).convert('L')
        sketch_array = np.array(sketch)
        return (sketch_array / 255.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state_variables()

        max_coord = self.canvas_size[0] - self.target_square_size
        y0 = self.np_random.integers(0, max_coord + 1)
        x0 = self.np_random.integers(0, max_coord + 1)

        self.target_sketch = np.full(self.canvas_size, 1.0, dtype=np.float32)
        self.target_sketch[y0:y0 + self.target_square_size, x0:x0 + self.target_square_size] = 0.0
        self.reward_map = calculate_reward_map(
            self.target_sketch,
            reward_on_target=self.reward_map_on_target,
            reward_near_target=self.reward_map_near_target,
            reward_far_target=self.reward_map_far_target,
            near_distance=self.reward_map_near_distance
        )
        # if len(self.target_sketches) == 1:
        #     self.target_sketch = self.target_sketches[0]
        # else:
        #     self.target_sketch = random.choice(self.target_sketches)

        self.cursor = find_starting_point(self.target_sketch)

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        dx, dy, is_pen_down, is_stop_action = _decode_action(action)

        self._update_agent_state(dx, dy, bool(is_pen_down), is_stop_action)
        terminated = is_stop_action

        canvas_before = self.canvas.copy()
        potential_affected_pixels = []
        brush_radius = self.brush_size // 2
        current_cursor = self.cursor
        if is_pen_down:
            y_start = max(0, current_cursor[1] - brush_radius)
            y_end = min(self.canvas_size[1], current_cursor[1] + brush_radius + 1)
            x_start = max(0, current_cursor[0] - brush_radius)
            x_end = min(self.canvas_size[0], current_cursor[0] + brush_radius + 1)
            self.canvas[y_start:y_end, x_start:x_end] = 0.0
            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    potential_affected_pixels.append((r, c))

        self._update_agent_state(dx, dy, is_pen_down, is_stop_action)
        reward = self._calculate_reward(
            terminated, False,
            is_pen_down, potential_affected_pixels, canvas_before
        )

        truncated = self.current_step >= self.max_steps or np.isclose(self.last_pixel_similarity, 1.0)

        #reward = self._calculate_reward(dx, dy, terminated, truncated)

        if terminated or truncated:
            self.episode_end = True
            self.last_pixel_similarity = calculate_pixel_similarity(self.canvas, self.target_sketch)

        observation = self._get_obs()
        #visualize_obs(observation)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _update_agent_state(self, dx, dy, is_pen_down, is_stop_action):
        if not is_stop_action:
            self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.canvas_size[0] - 1)
            self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.canvas_size[1] - 1)

            if self.pen_was_down and not is_pen_down:
                self.used_budgets = min(255, self.used_budgets + 1)

            self.is_pen_down = is_pen_down
            self.pen_was_down = self.is_pen_down

    def _calculate_reward(self, terminated, truncated,
                          is_pen_down, potential_affected_pixels, canvas_before):
        reward = 0.0

        if self.use_reward_map_reward:
            step_reward = 0.0
            newly_blackened_pixels_exist = False

            if is_pen_down and potential_affected_pixels:
                for r, c in potential_affected_pixels:
                    was_white = np.isclose(canvas_before[r, c], 1.0)
                    is_now_black = np.isclose(self.canvas[r, c], 0.0)

                    if was_white and is_now_black:
                        newly_blackened_pixels_exist = True
                        step_reward += self.reward_map[r, c]

            reward += step_reward if newly_blackened_pixels_exist else 0.0

        # if self.is_pen_down and np.isclose(self.canvas[self.cursor[1], self.cursor[0]], 1.0):
        #     self.canvas[self.cursor[1], self.cursor[0]] = 0.0
        #     # if self.use_local_reward_block:
        #     #     reward += calculate_density_cap_reward(self.canvas, self.target_sketch, self.cursor,
        #     #                                            self.local_reward_block_size)
        #     if not self.use_step_similarity_reward:
        #         is_correct = np.isclose(self.target_sketch[self.cursor[1], self.cursor[0]], 0.0)
        #         reward += 0.1 if is_correct else -0.1

        current_pixel_similarity = calculate_pixel_similarity(self.canvas, self.target_sketch)
        if self.use_step_similarity_reward:
            delta = current_pixel_similarity - self.last_pixel_similarity
            reward += self.similarity_weight * delta
            self.delta_similarity_history.append(delta)
        self.last_pixel_similarity = current_pixel_similarity

        if truncated or terminated:
            reward += self.last_pixel_similarity * self.similarity_weight
            #reward += self._calculate_final_reward() * self.r_stroke_hyper
            if self.use_stroke_reward and self.used_budgets > 0:
                reward += self.r_stroke_hyper / self.used_budgets * self.last_pixel_similarity
            if self.block_reward_scale > 0:
                self.block_similarity = calculate_block_reward(self.canvas, self.target_sketch, self.block_size)
                self.block_reward = self.block_similarity * self.block_reward_scale
                reward += self.block_reward

        self.step_rewards += reward
        return reward

    def _get_obs(self):
        pen_mask = np.full(self.canvas_size, 0.0, dtype=np.float32)
        y, x = self.cursor[1], self.cursor[0]
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            pen_mask[y, x] = 1.0

        budget_value = 0.0
        if self.dynamic_budget_channel:
            budget_value = (max(0, self.stroke_budget - self.used_budgets) / self.stroke_budget
                            if self.stroke_budget > 0 else 0.0)
        else:
            budget_value = self.stroke_budget / 255.0

        stroke_budget_channel = np.full(self.canvas_size, budget_value, dtype=np.float32)

        if not self.use_budget_channel:
            obs = np.stack([self.canvas.copy(), self.target_sketch.copy(), pen_mask], axis=0)
        else:
            obs = np.stack([self.canvas.copy(), self.target_sketch.copy(), pen_mask, stroke_budget_channel], axis=0)
        return obs

    def _get_info(self):
        info_dict = {
            "similarity": self.last_pixel_similarity,
            "used_budgets": self.used_budgets,
            "block_similarity": self.block_similarity,
            "block_reward": self.block_reward,
            "step_rewards": self.step_rewards,
        }
        if self.episode_end:
            info_dict["delta_similarity_history"] = self.delta_similarity_history
        return info_dict

    def _init_pygame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            WINDOW_WIDTH = (self.canvas_size[1] * 2 + 10) * self.render_scale
            WINDOW_HEIGHT = self.canvas_size[0] * self.render_scale
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Drawing Agent RL")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render(self):
        if self.render_mode != "human": return
        if self.window is None or self.clock is None: self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise Exception("Pygame window closed by user.")

        self.window.fill((105, 105, 105))
        self._draw_surface(self.target_sketch, (0, 0))
        self._draw_surface(self.canvas, ((self.canvas_size[0] + 10) * self.render_scale, 0))

        cursor_color = (255, 0, 0) if self.is_pen_down else (0, 0, 255)
        pygame.draw.circle(
            self.window, cursor_color,
            (self.canvas_size[1] * self.render_scale + 10 * self.render_scale + self.cursor[0] * self.render_scale,
             self.cursor[1] * self.render_scale),
            1 * self.render_scale
        )
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_surface(self, array, position):
        array_rgb = np.stack([(array.T * 255)] * 3, axis=-1).astype(np.uint8)
        surface = pygame.surfarray.make_surface(array_rgb)

        scaled_surface = pygame.transform.scale(
            surface,
            (self.canvas_size[1] * self.render_scale, self.canvas_size[0] * self.render_scale)
        )

        self.window.blit(scaled_surface, position)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None