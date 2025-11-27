import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import os
import random
import pygame
from envs.drawing_env.tools.image_process import find_starting_point, \
    calculate_block_reward, calculate_reward_map, calculate_dynamic_distance_map, calculate_metrics, calculate_f1_score, \
    visualize_obs
from collections import deque

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
    _episode_counter = 0

    def __init__(self, config=None):
        super(DrawingAgentEnv, self).__init__()
        self._init_config(config)
        self._init_state_variables()
        self.render_scale = 10

        self.action_space = spaces.Discrete(18) # 0-17 for movement, 18 for stop
        num_obs_channels = 1 + int(self.use_distance_map_obs) + int(self.use_budget_channel) + int(
            self.use_canvas_obs) + int(self.use_target_sketch_obs) + int(self.use_stroke_trajectory_obs) + int(self.use_combo_channel)

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
        self.use_budget_channel = config.get("use_budget_channel", False)
        self.use_combo = config.get("use_combo", False)
        self.use_triangles = config.get("use_triangles", False)
        self.combo_rate = config.get("combo_rate", 1.1)
        self.use_time_penalty = config.get("use_time_penalty", False)
        self.use_canvas_obs = config.get("use_canvas_obs", True)
        self.use_target_sketch_obs = config.get("use_target_sketch_obs", True)
        self.use_combo_channel = config.get("use_combo_channel", False)
        self.max_combo_normalization = 50.0
        self.use_stroke_trajectory_obs = config.get("use_stroke_trajectory_obs", False)

        self.use_mvg_penalty_compensation = config.get("use_mvg_penalty_compensation", False)
        self.mvg_penalty_window_size = self.max_steps // 5

        self.use_distance_map_obs = config.get("use_distance_map_obs", False)
        max_dist = np.sqrt((self.canvas_size[0] - 1) ** 2 + (self.canvas_size[1] - 1) ** 2)
        self.max_obs_distance = max(max_dist, 1.0)

        self.use_dynamic_distance_map_reward = config.get("use_dynamic_distance_map_reward", False)
        self.navigation_reward_scale = config.get("navigation_reward_scale", 0.05)

        self.num_rectangles = config.get("num_rectangles", 2)
        self.rect_min_width = config.get("rect_min_width", 5)
        self.rect_max_width = config.get("rect_max_width", 15)
        self.rect_min_height = config.get("rect_min_height", 5)
        self.rect_max_height = config.get("rect_max_height", 15)
        self.penalty_scale_threshold = config.get("penalty_scale_threshold", 0.9)

        self.brush_size = config.get("brush_size", 1)
        self.target_square_size = config.get("target_square_size", 15)
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
        self.similarity_weight = config.get("similarity_weight", 0.0)

        self.target_data = config.get("precalculated_data", None)

        if self.target_data is None:
            print("[Warning] No 'precalculated_data' found. Falling back to slow loading inside env...")
            self.target_sketches_path = config.get("target_sketches_path", None)
            self.specific_sketch_file = config.get("specific_sketch_file", None)
            self.target_data = self._load_target_sketches()

        if not self.target_data:
            raise ValueError("No target data was loaded or provided!")

        self.step_debug_path = config.get("step_debug_path", None)
        self.episode_save_limit = config.get("episode_save_limit", 1000)
        self.current_episode_num = 0

        self.recall_bonus = config.get("recall_bonus", 0.0)
        self.f1_scalar = config.get("f1_scalar", 0.0)

    def _init_state_variables(self):
        self.current_step = 0
        self.canvas = np.full(self.canvas_size, 1.0, dtype=np.float32)
        self.used_budgets = 0
        self.is_pen_down = False
        self.pen_was_down = False
        self.episode_end = False
        self.navigation_reward = 0
        self.current_combo = 0
        self.episode_combo_log = []
        self.current_stroke_trajectory = []

        self._current_tp = 0
        self._current_tn = 0
        self._current_fp = 0
        self._current_fn = 0

        self.last_pixel_similarity = 0.0
        self.last_iou_similarity = 0.0
        self.last_balanced_accuracy = 0.0
        self.last_recall_black = 0.0
        self.last_recall_white = 0.0
        self.last_precision_black = 0.0

        self.block_similarity = 0
        self.block_reward = 0
        self.step_rewards = 0
        self.delta_similarity_history = []
        self.target_sketch = None
        self.cursor = [0, 0]
        self.episode_total_painted = 0
        self.episode_correctly_painted = 0

        self.penalty_history = deque(maxlen=self.mvg_penalty_window_size)
        self.current_mvg_penalty = 0.0

        self.current_episode_step_data = []

        self.dynamic_distance_map = None
        self.last_distance = 0.0
        self.needs_distance_map_update = True

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

        DrawingAgentEnv._episode_counter += 1
        self.current_episode_num = DrawingAgentEnv._episode_counter
        self._init_state_variables()
        self.correct_rewards = 0
        if self.use_triangles:
            self.target_sketch = np.full(self.canvas_size, 1.0, dtype=np.float32)
            for _ in range(self.num_rectangles):
                rect_width = self.np_random.integers(self.rect_min_width, self.rect_max_width + 1)
                rect_height = self.np_random.integers(self.rect_min_height, self.rect_max_height + 1)

                max_x0 = self.canvas_size[0] - rect_width
                max_y0 = self.canvas_size[1] - rect_height

                x0 = self.np_random.integers(0, max_x0 + 1)
                y0 = self.np_random.integers(0, max_y0 + 1)

                self.target_sketch[y0: y0 + rect_height, x0: x0 + rect_width] = 0.0
            self.reward_map = calculate_reward_map(
                self.target_sketch,
                reward_on_target=self.reward_map_on_target,
                reward_near_target=self.reward_map_near_target,
                reward_far_target=self.reward_map_far_target,
                near_distance=self.reward_map_near_distance
            )

            self._update_dynamic_distance_map()
        else:
            if isinstance(self.target_data[0], tuple):
                chosen_data = random.choice(self.target_data)
                self.target_sketch = chosen_data[0]
                self.reward_map = chosen_data[1]
                self.dynamic_distance_map = chosen_data[2]
            else:
                if len(self.target_data) == 1:
                    self.target_sketch = self.target_data[0]
                else:
                    self.target_sketch = random.choice(self.target_data)

                self.reward_map = calculate_reward_map(
                        self.target_sketch,
                        reward_on_target=self.reward_map_on_target,
                        reward_near_target=self.reward_map_near_target,
                        reward_far_target=self.reward_map_far_target,
                        near_distance=self.reward_map_near_distance
                    )
                self._update_dynamic_distance_map()

        self.cursor = find_starting_point(self.target_sketch)
        if self.use_dynamic_distance_map_reward:
            self.last_distance = self.dynamic_distance_map[self.cursor[1], self.cursor[0]]

        target_is_black = np.isclose(self.target_sketch, 0.0)
        target_is_white = np.isclose(self.target_sketch, 1.0)
        self._current_tp = 0
        self._current_fp = 0
        self._current_tn = np.sum(target_is_white)
        self._current_fn = np.sum(target_is_black)

        self.last_recall_black = 0.0
        self.last_recall_white = 1.0 if self._current_tn > 0 else 0.0
        self.last_precision_black = 0.0
        self.last_f1_score = 0.0

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        dx, dy, is_pen_down, is_stop_action = _decode_action(action)

        self._update_agent_state(dx, dy, bool(is_pen_down), is_stop_action)
        terminated = is_stop_action

        correct_new_pixels = []
        repeated_correct_pixels = []
        if is_pen_down and not terminated:
            current_cursor = self.cursor
            brush_radius = self.brush_size // 2

            y_start = max(0, current_cursor[1] - brush_radius)
            y_end = min(self.canvas_size[1], current_cursor[1] + brush_radius + 1)
            x_start = max(0, current_cursor[0] - brush_radius)
            x_end = min(self.canvas_size[0], current_cursor[0] + brush_radius + 1)

            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):
                        if np.isclose(self.target_sketch[r, c], 0.0):
                            correct_new_pixels.append((r, c))
                            self._current_tp += 1
                            self._current_fn -= 1
                        else:
                            self._current_fp += 1
                            self._current_tn -= 1
                    else:
                        if np.isclose(self.target_sketch[r, c], 0.0):
                            repeated_correct_pixels.append((r, c))

            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):
                        self.episode_total_painted += 1
                        self.canvas[r, c] = 0.0

        current_recall_black, current_recall_white, current_precision_black, current_pixel_similarity  = calculate_metrics(
            self._current_tp, self._current_fp, self._current_tn, self._current_fn,
            self.canvas.size)

        self.last_pixel_similarity = current_pixel_similarity
        self.last_recall_black = current_recall_black
        self.last_recall_white = current_recall_white
        self.last_precision_black = current_precision_black
        current_f1_score = calculate_f1_score(self.last_precision_black, self.last_recall_black)
        truncated = self.current_step >= self.max_steps or np.isclose(self.last_recall_black, 1.0)

        reward = self._calculate_reward(
            terminated, truncated,
            is_pen_down,
            correct_new_pixels,
            repeated_correct_pixels,
            current_f1_score)
        self.last_f1_score = current_f1_score

        if terminated or truncated:
            if self.current_combo > 0:
                self.episode_combo_log.append(self.current_combo)
            self.episode_end = True
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

            if self.use_stroke_trajectory_obs:
                if is_pen_down:
                    if not self.pen_was_down:
                        self.current_stroke_trajectory = []

                    self.current_stroke_trajectory.append(tuple(self.cursor))
                else:
                    self.current_stroke_trajectory = []

            if self.pen_was_down and not is_pen_down:
                self.used_budgets = min(255, self.used_budgets + 1)

            self.is_pen_down = is_pen_down
            self.pen_was_down = self.is_pen_down

    def _calculate_reward(self, terminated, truncated,
                          is_pen_down,correct_new_pixels, repeated_new_pixels, current_f1_score):
        reward = 0.0 if not self.use_time_penalty else -0.001
        drawing_reward = 0.0
        negative_reward_this_step = 0.0
        if is_pen_down:
            num_correct = len(correct_new_pixels)
            num_repeated = len(repeated_new_pixels)
            self.episode_correctly_painted += num_correct
            if num_correct > 0:
                positive_reward_this_step = 0.0
                for r, c in correct_new_pixels:
                    positive_reward_this_step += self.reward_map[r, c]

                self.current_combo += 1
                if self.use_combo:
                    if self.combo_rate < 1.0:
                        positive_reward_this_step *= (1 + self.combo_rate * self.current_combo)
                    else:
                        positive_reward_this_step *= (self.combo_rate ** self.current_combo)

                drawing_reward = positive_reward_this_step
                self.correct_rewards += positive_reward_this_step

            elif num_correct == 0 and num_repeated > 0:
                if self.current_combo > 0:
                    self.episode_combo_log.append(self.current_combo)
                self.current_combo = 0
            elif num_correct == 0 and num_repeated == 0:
                if self.current_combo > 0:
                    self.episode_combo_log.append(self.current_combo)
                self.current_combo = 0
                current_penalty_scale = 0.0
                if self.penalty_scale_threshold > 0: # if negative, then no penalty
                    if self.penalty_scale_threshold <= self.last_recall_black < 1.0:
                        current_penalty_scale = self.last_precision_black
                    elif self.last_recall_black >= 1.0 or self.penalty_scale_threshold > 1.0:
                        current_penalty_scale = 1.0

                negative_reward_this_step = self.reward_map_far_target * current_penalty_scale

                if negative_reward_this_step < 0:
                    self.penalty_history.append(negative_reward_this_step)
                else:
                    self.penalty_history.append(0)

                if len(self.penalty_history) > 0:
                    self.current_mvg_penalty = sum(self.penalty_history) / len(self.penalty_history)
                else:
                    self.current_mvg_penalty = 0.0

                if self.use_mvg_penalty_compensation and self.last_recall_black >= self.penalty_scale_threshold:
                    negative_reward_this_step += -self.current_mvg_penalty

                drawing_reward = negative_reward_this_step

            if num_correct > 0 and self.use_dynamic_distance_map_reward:
                self._update_dynamic_distance_map()
        else:
            if self.current_combo > 0:
                self.episode_combo_log.append(self.current_combo)
            self.current_combo = 0
            if self.use_dynamic_distance_map_reward:
                current_distance = self.dynamic_distance_map[self.cursor[1], self.cursor[0]]
                self.navigation_reward = (self.last_distance - current_distance) * self.navigation_reward_scale
                reward += self.navigation_reward

                self.last_distance = current_distance

        if self.f1_scalar > 0:
            delta_f1 = current_f1_score - self.last_f1_score
            if delta_f1 >= 0:
                reward += delta_f1 * self.f1_scalar
            else:
                reward += delta_f1 * (self.f1_scalar / 10)
        else:
            reward += drawing_reward

        #if truncated or terminated:
            #reward += self.last_precision_black * self.similarity_weight
            #if np.isclose(self.last_recall_black, 1.0):
                #reward += self.recall_bonus
            #reward += self._calculate_final_reward() * self.r_stroke_hyper
            # if self.use_stroke_reward and self.used_budgets > 0:
            #     reward += self.r_stroke_hyper / self.used_budgets * self.last_recall_black
            # if self.block_reward_scale > 0:
            #     self.block_similarity = calculate_block_reward(self.canvas, self.target_sketch, self.block_size)
            #     self.block_reward = self.block_similarity * self.block_reward_scale
            #     reward += self.block_reward

        self.step_rewards += reward
        return reward

    def _update_dynamic_distance_map(self):
        self.dynamic_distance_map = calculate_dynamic_distance_map(self.target_sketch, self.canvas)
        self.needs_distance_map_update = False

    def _get_obs(self):
        if not hasattr(self, "_obs"):
            self._obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        ch_idx = 0
        if self.use_canvas_obs:
            self._obs[ch_idx][:] = self.canvas
            ch_idx += 1

        if self.use_target_sketch_obs:
            if self.current_step == 0:
                self._obs[ch_idx] = self.target_sketch.astype(np.float32)
            ch_idx += 1

        pen_mask = self._obs[ch_idx]
        pen_mask.fill(0.0)
        y, x = self.cursor[1], self.cursor[0]
        #brush_radius = self.brush_size // 2
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            # y_start = max(0, y - brush_radius)
            # y_end = min(y + brush_radius + 1, self.canvas_size[0])
            # x_start = max(0, x - brush_radius)
            # x_end = min(x + brush_radius + 1, self.canvas_size[1])
            #pen_mask[y_start:y_end, x_start:x_end] = 1.0
            pen_mask[y, x] = 1.0

        ch_idx += 1

        if self.use_distance_map_obs:
            normalized_dist_map = self.dynamic_distance_map / self.max_obs_distance
            np.clip(normalized_dist_map, 0.0, 1.0, out=self._obs[ch_idx])
            ch_idx += 1

        if self.use_budget_channel:
            if self.dynamic_budget_channel:
                budget_value = (max(0, self.stroke_budget - self.used_budgets) / self.stroke_budget
                                if self.stroke_budget > 0 else 0.0)
                self._obs[ch_idx][:] = budget_value
            elif self.current_step == 0:
                budget_value = self.stroke_budget / 255.0
                self._obs[ch_idx] = budget_value

        if self.use_combo_channel:
            combo_value = min(self.current_combo / self.max_combo_normalization, 1.0)
            self._obs[ch_idx][:] = combo_value
            ch_idx += 1

        if self.use_stroke_trajectory_obs:
            traj_map = self._obs[ch_idx]
            traj_map.fill(0.0)

            brush_radius = self.brush_size // 2
            H, W = self.canvas_size[0], self.canvas_size[1]

            for cx, cy in self.current_stroke_trajectory:
                y_start = max(0, cy - brush_radius)
                y_end = min(H, cy + brush_radius + 1)
                x_start = max(0, cx - brush_radius)
                x_end = min(W, cx + brush_radius + 1)

                traj_map[y_start:y_end, x_start:x_end] = 1.0

            ch_idx += 1

        return self._obs

    def _get_info(self):
        info_dict = {
            "pixel_similarity": self.last_pixel_similarity,
            "recall_black": self.last_recall_black,
            "recall_white": self.last_recall_white,
            "used_budgets": self.used_budgets,
            "block_similarity": self.block_similarity,
            "block_reward": self.block_reward,
            "step_rewards": self.step_rewards,
            "total_painted": self.episode_total_painted,
            "correctly_painted": self.episode_correctly_painted,
            "navigation_reward": self.navigation_reward,
            "combo_count": self.current_combo,
            "precision": self.last_precision_black,
            "f1_score": self.last_f1_score,
            "episode_combo_log": self.episode_combo_log,
        }
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

    def _save_step_data(self, action=None, reward=None, is_reset=False):
        if not self.step_debug_path or self.current_episode_num > self.episode_save_limit:
            return

        try:
            episode_dir = os.path.join(self.step_debug_path, f"episode_{self.current_episode_num:04d}")
            os.makedirs(episode_dir, exist_ok=True)

            step_num = self.current_step if not is_reset else 0
            if step_num >= self.max_steps:
                img_path = os.path.join(episode_dir, f"result.png")
                canvas_img_array = (self.canvas * 255).astype(np.uint8)
                img = Image.fromarray(canvas_img_array, 'L')
                img.save(img_path)

            if is_reset:
                target_img_path = os.path.join(episode_dir, "_target.png")
                target_img_array = (self.target_sketch * 255).astype(np.uint8)
                target_img = Image.fromarray(target_img_array, 'L')
                target_img.save(target_img_path)

            step_log = {
                "step": step_num,
                "action": int(action) if action is not None else None,
                "reward": float(reward) if reward is not None else None,
            }
            self.current_episode_step_data.append(step_log)
        except Exception as e:
            print(
                f"[StepLogger] Error saving step data for ep {self.current_episode_num}, step {self.current_step}: {e}")

    def _save_step_log_file(self):
        if not self.step_debug_path or self.current_episode_num > self.episode_save_limit:
            return

        if not self.current_episode_step_data:
            print(f"[StepLogger] No step data to save for episode {self.current_episode_num}")
            return

        episode_dir = os.path.join(self.step_debug_path, f"episode_{self.current_episode_num:04d}")
        log_path = os.path.join(episode_dir, "_step_log.json")

        try:
            with open(log_path, 'w') as f:
                json.dump(self.current_episode_step_data, f, indent=4)
            print(f"[StepLogger] Saved step log for episode {self.current_episode_num} to {log_path}")
        except Exception as e:
            print(f"[StepLogger] Error saving JSON log for episode {self.current_episode_num}: {e}")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None