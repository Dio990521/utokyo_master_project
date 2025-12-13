import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import os
import random
import pygame
from envs.drawing_env.tools.image_process import (
    find_starting_point,
    calculate_f1_score, calculate_metrics, get_active_endpoints
)
from collections import deque


def _decode_action(action):
    # 0-8: Move + Pen Up
    # 9-17: Move + Pen Down
    # 18: Jump
    is_jump = False
    if action == 18:
        is_jump = True
        return 0, 0, 0, 0, is_jump

    is_pen_down = action >= 9
    sub_action = action % 9
    dx = (sub_action % 3) - 1
    dy = (sub_action // 3) - 1
    return dx, dy, int(is_pen_down), 0, is_jump


def _decode_multi_discrete_action(action):
    move_idx = action[0]
    pen_idx = action[1]

    is_jump = False
    if move_idx == 9:
        is_jump = True
        return 0, 0, bool(pen_idx), 0, is_jump

    dx = (move_idx % 3) - 1
    dy = (move_idx // 3) - 1

    is_pen_down = bool(pen_idx)
    return dx, dy, is_pen_down, 0, is_jump


class DrawingAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    _episode_counter = 0

    def __init__(self, config=None):
        super(DrawingAgentEnv, self).__init__()
        self._init_config(config)
        self._init_state_variables()
        self.render_scale = 10

        self.use_jump = config.get("use_jump", False)

        if self.use_multi_discrete:
            move_dim = 10 if self.use_jump else 9
            self.action_space = spaces.MultiDiscrete([move_dim, 2])
        else:
            act_dim = 19 if self.use_jump else 18
            self.action_space = spaces.Discrete(act_dim)

        num_obs_channels = (
                1 +  # Pen Mask (Always included in logic below)
                int(self.use_budget_channel) +
                int(self.use_canvas_obs) +
                int(self.use_target_sketch_obs) +
                int(self.use_stroke_trajectory_obs) +
                int(self.use_combo_channel)
        )

        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(num_obs_channels, *self.canvas_size), dtype=np.float32
        )

        self.window = None
        self.clock = None

    def _init_config(self, config):
        config = config or {}
        self.canvas_size = config.get("canvas_size", [32, 32])
        self.max_steps = config.get("max_steps", 1000)
        self.render_mode = config.get("render_mode", None)

        # Budget & Combo
        self.stroke_budget = config.get("stroke_budget", 1)
        self.dynamic_budget_channel = config.get("dynamic_budget_channel", False)
        self.use_budget_channel = config.get("use_budget_channel", False)
        self.use_combo = config.get("use_combo", False)
        self.use_combo_channel = config.get("use_combo_channel", False)
        self.combo_rate = config.get("combo_rate", 1.1)
        self.max_combo_normalization = 50.0

        # Obs Flags
        self.use_canvas_obs = config.get("use_canvas_obs", True)
        self.use_target_sketch_obs = config.get("use_target_sketch_obs", True)
        self.use_stroke_trajectory_obs = config.get("use_stroke_trajectory_obs", False)
        self.use_distance_map_obs = config.get("use_distance_map_obs", False)

        self.use_multi_discrete = config.get("use_multi_discrete", False)

        # Penalties & Rewards
        self.reward_correct = config.get("reward_correct", 0.1)
        self.reward_wrong = config.get("reward_wrong", -0.01)
        self.use_time_penalty = config.get("use_time_penalty", False)
        self.use_mvg_penalty_compensation = config.get("use_mvg_penalty_compensation", False)
        self.mvg_penalty_window_size = self.max_steps // 5
        self.penalty_scale_threshold = config.get("penalty_scale_threshold", 0.9)

        self.use_dynamic_distance_map_reward = config.get("use_dynamic_distance_map_reward", False)
        self.navigation_reward_scale = config.get("navigation_reward_scale", 0.05)

        self.reward_map_on_target = config.get("reward_map_on_target", 0.1)
        self.reward_map_near_target = config.get("reward_map_near_target", 0.0)
        self.reward_map_far_target = config.get("reward_map_far_target", -0.1)
        self.reward_map_near_distance = config.get("reward_map_near_distance", 2)

        self.similarity_weight = config.get("similarity_weight", 0.0)
        self.use_distance_reward = config.get("use_distance_reward", False)
        self.dist_scale = config.get("distance_reward_scale", 0.05)

        self.use_skeleton_guidance = config.get("use_skeleton_guidance", False)
        self.use_rook_move = config.get("use_rook_move", False)

        # Drawing Config
        self.brush_size = config.get("brush_size", 1)
        self.use_triangles = config.get("use_triangles", False)
        self.num_rectangles = config.get("num_rectangles", 2)
        self.rect_min_width = config.get("rect_min_width", 5)
        self.rect_max_width = config.get("rect_max_width", 15)
        self.rect_min_height = config.get("rect_min_height", 5)
        self.rect_max_height = config.get("rect_max_height", 15)

        # Data Configs
        self.target_sketches_path = config.get("target_sketches_path", None)
        self.specific_sketch_file = config.get("specific_sketch_file", None)
        self.sketch_file_list = []
        if self.specific_sketch_file and os.path.exists(self.specific_sketch_file):
            # Validation mode: single file
            self.sketch_file_list = [self.specific_sketch_file]
        elif self.target_sketches_path and os.path.exists(self.target_sketches_path):
            # Training mode: list all files
            self.sketch_file_list = [
                os.path.join(self.target_sketches_path, f)
                for f in os.listdir(self.target_sketches_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not self.sketch_file_list:
                raise ValueError(f"No image files found in {self.target_sketches_path}")
        elif not self.use_triangles:
            raise ValueError("No target sketch path or specific file provided!")

        self.step_debug_path = config.get("step_debug_path", None)
        self.episode_save_limit = config.get("episode_save_limit", 1000)
        self.current_episode_num = 0

    def _init_state_variables(self):
        self.current_step = 0
        self.canvas = np.full(self.canvas_size, 1.0, dtype=np.float32)
        self.used_budgets = 0
        self.is_pen_down = False
        self.pen_was_down = False
        self.episode_end = False
        self.navigation_reward = 0
        self.current_combo = 0
        self.combo_sustained_on_repeat = 0
        self.episode_combo_log = []
        self.episode_combo_sustained_on_repeat_log = []
        self.current_stroke_trajectory = []

        self._current_tp = 0
        self._current_tn = 0
        self._current_fp = 0
        self._current_fn = 0

        self.last_pixel_similarity = 0.0
        self.last_recall_black = 0.0
        self.last_recall_white = 0.0
        self.last_precision_black = 0.0
        self.last_f1_score = 0.0

        self.step_rewards = 0
        self.target_sketch = None
        self.cursor = [0, 0]
        self.episode_total_painted = 0
        self.episode_correctly_painted = 0

        self.penalty_history = deque(maxlen=self.mvg_penalty_window_size)
        self.current_mvg_penalty = 0.0
        self.current_episode_step_data = []

        self.dynamic_distance_map = None
        self.last_distance = 0.0

        self.episode_base_reward = 0.0
        self.episode_combo_bonus = 0.0
        self.episode_negative_reward = 0.0

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
        sketch = Image.open(filepath).convert('L')
        sketch_array = np.array(sketch)
        return (sketch_array / 255.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        DrawingAgentEnv._episode_counter += 1
        self.current_episode_num = DrawingAgentEnv._episode_counter
        self._init_state_variables()
        self.correct_rewards = 0

        # Load Target
        if self.use_triangles:
            self.target_sketch = np.full(self.canvas_size, 1.0, dtype=np.float32)
            # ... (Triangle generation logic remains same) ...
            for _ in range(self.num_rectangles):
                rect_width = self.np_random.integers(self.rect_min_width, self.rect_max_width + 1)
                rect_height = self.np_random.integers(self.rect_min_height, self.rect_max_height + 1)
                max_x0 = self.canvas_size[0] - rect_width
                max_y0 = self.canvas_size[1] - rect_height
                x0 = self.np_random.integers(0, max_x0 + 1)
                y0 = self.np_random.integers(0, max_y0 + 1)
                self.target_sketch[y0: y0 + rect_height, x0: x0 + rect_width] = 0.0

        else:
            if not self.sketch_file_list:
                raise ValueError("Sketch file list is empty!")
            chosen_file = random.choice(self.sketch_file_list)
            self.target_sketch = self._load_sketch_from_path(chosen_file)

        self.cursor = find_starting_point(self.target_sketch)
        if self.use_dynamic_distance_map_reward:
            self.last_distance = self.dynamic_distance_map[self.cursor[1], self.cursor[0]]

        target_is_black = np.isclose(self.target_sketch, 0.0)
        target_is_white = np.isclose(self.target_sketch, 1.0)
        self._current_tp = 0
        self._current_fp = 0
        self._current_tn = np.sum(target_is_white)
        self._current_fn = np.sum(target_is_black)

        self.last_recall_white = 1.0 if self._current_tn > 0 else 0.0

        if self.use_distance_reward:
            _, self.last_min_dist = self._find_nearest_target_pixel()

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), self._get_info()

    def _perform_rook_move(self, dx, dy):
        if dx == 0 and dy == 0:
            return

        curr_x, curr_y = self.cursor[0], self.cursor[1]
        width, height = self.canvas_size[1], self.canvas_size[0]

        found_target = False
        target_x, target_y = curr_x, curr_y

        max_steps = max(width, height)

        temp_x, temp_y = curr_x, curr_y
        for _ in range(max_steps):
            temp_x += dx
            temp_y += dy

            if not (0 <= temp_x < width and 0 <= temp_y < height):
                break

            if (self.canvas[temp_y, temp_x] - self.target_sketch[temp_y, temp_x]) > 0.05:
                target_x, target_y = temp_x, temp_y
                found_target = True
                break

        if found_target:
            mid_x = (curr_x + target_x) // 2
            mid_y = (curr_y + target_y) // 2
            if mid_x == curr_x and mid_y == curr_y:
                self.cursor[0] = target_x
                self.cursor[1] = target_y
            else:
                self.cursor[0] = mid_x
                self.cursor[1] = mid_y
        else:
            self._update_agent_state(dx, dy, False, 0)

    def _find_nearest_target_pixel(self):
        target_ink_mask = self.target_sketch < 0.5
        canvas_ink_mask = self.canvas < 0.5

        overlap_mask = target_ink_mask & canvas_ink_mask

        remaining_target_mask = target_ink_mask & (~overlap_mask)

        target_indices_yx = np.argwhere(remaining_target_mask)

        if len(target_indices_yx) == 0:
            return self.cursor, self.canvas_size[0]

        current_pos_yx = np.array([self.cursor[1], self.cursor[0]])

        dists = np.linalg.norm(target_indices_yx - current_pos_yx, axis=1)
        min_idx = np.argmin(dists)
        nearest_pixel_yx = target_indices_yx[min_idx]

        return [nearest_pixel_yx[1], nearest_pixel_yx[0]], np.min(dists)

    def step(self, action):
        self.current_step += 1
        if self.use_multi_discrete:
            dx, dy, is_pen_down, is_stop_action, is_jump = _decode_multi_discrete_action(action)
        else:
            dx, dy, is_pen_down, is_stop_action, is_jump = _decode_action(action)
        # is_jump = True
        # is_pen_down = True
        if self.use_jump and is_jump:
            new_pos = self._find_nearest_target_pixel()
            self.cursor[0] = np.clip(new_pos[0], 0, self.canvas_size[0] - 1)
            self.cursor[1] = np.clip(new_pos[1], 0, self.canvas_size[1] - 1)
            self.is_pen_down = bool(is_pen_down)
            self.pen_was_down = False
        else:
            if not is_pen_down and self.use_rook_move:
                self._perform_rook_move(dx, dy)
            else:
                self._update_agent_state(dx, dy, bool(is_pen_down), 0)
        terminated = is_stop_action

        dist_reward = 0.0
        if self.use_distance_reward:
            _, dist_before_paint = self._find_nearest_target_pixel()
            dist_reward = (self.last_min_dist - dist_before_paint) * self.dist_scale
        correct_new_pixels = []
        repeated_correct_pixels = []

        if is_pen_down and not terminated:
            current_cursor = self.cursor
            brush_radius = self.brush_size // 2

            # Calculate affected area
            y_start = max(0, current_cursor[1] - brush_radius)
            y_end = min(self.canvas_size[1], current_cursor[1] + brush_radius + 1)
            x_start = max(0, current_cursor[0] - brush_radius)
            x_end = min(self.canvas_size[0], current_cursor[0] + brush_radius + 1)

            # Check pixels
            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):  # If canvas is white (newly painted)
                        if np.isclose(self.target_sketch[r, c], 0.0):  # And target is black
                            correct_new_pixels.append((r, c))
                            self._current_tp += 1
                            self._current_fn -= 1
                        else:  # Target is white (wrong)
                            self._current_fp += 1
                            self._current_tn -= 1
                    else:  # Canvas is already black
                        if np.isclose(self.target_sketch[r, c], 0.0):
                            repeated_correct_pixels.append((r, c))

            # Apply paint
            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):
                        self.episode_total_painted += 1
                        self.canvas[r, c] = 0.0

        # Metrics
        current_recall_black, current_recall_white, current_precision_black, current_pixel_similarity = calculate_metrics(
            self._current_tp, self._current_fp, self._current_tn, self._current_fn, self.canvas.size
        )

        self.last_pixel_similarity = current_pixel_similarity
        self.last_recall_black = current_recall_black
        self.last_recall_white = current_recall_white
        self.last_precision_black = current_precision_black

        current_f1_score = calculate_f1_score(self.last_precision_black, self.last_recall_black)

        truncated = self.current_step >= self.max_steps or np.isclose(self.last_recall_black, 1.0)

        # Reward Calculation
        reward = self._calculate_reward(
            terminated, truncated,
            is_pen_down,
            correct_new_pixels,
            repeated_correct_pixels,
            current_f1_score
        )
        self.last_f1_score = current_f1_score

        if self.use_distance_reward:
            _, dist_after_paint = self._find_nearest_target_pixel()

            self.last_min_dist = dist_after_paint
            reward += dist_reward

        if terminated or truncated:
            if self.current_combo > 0:
                self.episode_combo_log.append(self.current_combo)
            if self.combo_sustained_on_repeat > 0:
                self.episode_combo_sustained_on_repeat_log.append(self.combo_sustained_on_repeat)
            self.episode_end = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    # def _calc_min_dist_to_unfinished(self):
    #     nearest_target_xy = np.array(self._find_nearest_target_pixel())
    #
    #     current_pos_xy = np.array(self.cursor)
    #     dist = np.linalg.norm(nearest_target_xy - current_pos_xy)
    #
    #     return dist

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
                          is_pen_down, correct_new_pixels, repeated_new_pixels, current_f1_score):
        reward = 0.0 if not self.use_time_penalty else -0.001
        drawing_reward = 0.0
        base_reward_part = 0.0
        bonus_reward_part = 0.0
        if is_pen_down:
            num_correct = len(correct_new_pixels)
            num_repeated = len(repeated_new_pixels)
            self.episode_correctly_painted += num_correct

            if num_correct > 0:
                # Correctly painted new pixels
                positive_reward_this_step = 0.0
                for r, c in correct_new_pixels:
                    positive_reward_this_step += self.reward_correct

                base_reward_part = positive_reward_this_step
                self.current_combo += 1
                self.combo_sustained_on_repeat += 1
                if self.use_combo:
                    if self.combo_rate < 1.0:
                        positive_reward_this_step *= (1 + self.combo_rate * self.combo_sustained_on_repeat)
                    else:
                        positive_reward_this_step *= (self.combo_rate ** self.combo_sustained_on_repeat)

                bonus_reward_part = positive_reward_this_step - base_reward_part
                drawing_reward = positive_reward_this_step
                self.correct_rewards += positive_reward_this_step

            elif num_correct == 0 and num_repeated > 0:
                if self.current_combo > 0:
                    self.episode_combo_log.append(self.current_combo)
                self.current_combo = 0
                self.combo_sustained_on_repeat += 1
                current_penalty_scale = 0.0
                if self.penalty_scale_threshold > 0:
                    if self.penalty_scale_threshold <= self.last_recall_black < 1.0:
                        current_penalty_scale = self.last_precision_black
                    elif self.last_recall_black >= 1.0 or self.penalty_scale_threshold > 1.0:
                        current_penalty_scale = 1.0

                negative_reward_this_step = self.reward_wrong * 0.5 * current_penalty_scale
                self.episode_negative_reward += negative_reward_this_step
                drawing_reward = negative_reward_this_step
            elif num_correct == 0 and num_repeated == 0:
                # Penalty logic for drawing on background
                if self.current_combo > 0:
                    self.episode_combo_log.append(self.current_combo)
                if self.combo_sustained_on_repeat > 0:
                    self.episode_combo_sustained_on_repeat_log.append(self.combo_sustained_on_repeat)
                self.current_combo = 0
                self.combo_sustained_on_repeat = 0

                current_penalty_scale = 0.0
                if self.penalty_scale_threshold > 0:
                    if self.penalty_scale_threshold <= self.last_recall_black < 1.0:
                        current_penalty_scale = self.last_precision_black
                    elif self.last_recall_black >= 1.0 or self.penalty_scale_threshold > 1.0:
                        current_penalty_scale = 1.0

                negative_reward_this_step = self.reward_wrong * current_penalty_scale
                self.episode_negative_reward += negative_reward_this_step
                drawing_reward = negative_reward_this_step

        else:
            # Pen Up
            if self.current_combo > 0:
                self.episode_combo_log.append(self.current_combo)
            if self.combo_sustained_on_repeat > 0:
                self.episode_combo_sustained_on_repeat_log.append(self.combo_sustained_on_repeat)
            self.current_combo = 0
            self.combo_sustained_on_repeat = 0

            if self.use_dynamic_distance_map_reward:
                current_distance = self.dynamic_distance_map[self.cursor[1], self.cursor[0]]
                self.navigation_reward = (self.last_distance - current_distance) * self.navigation_reward_scale
                reward += self.navigation_reward
                self.last_distance = current_distance

        reward += drawing_reward
        self.episode_base_reward += base_reward_part
        self.episode_combo_bonus += bonus_reward_part
        self.step_rewards += reward
        return reward

    def _get_obs(self):
        if not hasattr(self, "_obs"):
            self._obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        ch_idx = 0

        # 1. Canvas
        if self.use_canvas_obs:
            self._obs[ch_idx][:] = self.canvas
            ch_idx += 1

        # 2. Target Sketch
        if self.use_target_sketch_obs:
            if self.current_step == 0:
                self._obs[ch_idx] = self.target_sketch.astype(np.float32)
            ch_idx += 1

        # 3. Pen Mask
        pen_layer = self._obs[ch_idx]
        pen_layer.fill(0.0)

        if self.use_skeleton_guidance:
            active_endpoints = get_active_endpoints(self.target_sketch, self.canvas)
            if active_endpoints is not None:
                pen_layer[active_endpoints] = 0.5

        y, x = self.cursor[1], self.cursor[0]
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            pen_layer[y, x] = 1.0

        ch_idx += 1

        # 5. Budget
        if self.use_budget_channel:
            if self.dynamic_budget_channel:
                budget_value = (max(0, self.stroke_budget - self.used_budgets) / self.stroke_budget
                                if self.stroke_budget > 0 else 0.0)
                self._obs[ch_idx][:] = budget_value
            elif self.current_step == 0:
                budget_value = self.stroke_budget / 255.0
                self._obs[ch_idx] = budget_value
            ch_idx += 1  # Added explicit increment just in case logic continues

        # 6. Combo
        if self.use_combo_channel:
            combo_value = min(self.current_combo / self.max_combo_normalization, 1.0)
            self._obs[ch_idx][:] = combo_value
            ch_idx += 1

        # 7. Stroke Trajectory
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
            "step_rewards": self.step_rewards,
            "total_painted": self.episode_total_painted,
            "correctly_painted": self.episode_correctly_painted,
            "navigation_reward": self.navigation_reward,
            "combo_count": self.current_combo,
            "precision": self.last_precision_black,
            "f1_score": self.last_f1_score,
            "episode_combo_log": self.episode_combo_log,
            "episode_base_reward": self.episode_base_reward,
            "episode_combo_bonus": self.episode_combo_bonus,
            "combo_sustained": self.episode_combo_sustained_on_repeat_log,
            "negative_reward": self.episode_negative_reward
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

        right_panel_x_start = (self.canvas_size[1] + 10) * self.render_scale

        self._draw_surface(self.canvas, (right_panel_x_start, 0))

        cursor_color = (255, 0, 0) if self.is_pen_down else (0, 0, 255)

        pygame.draw.circle(
            self.window, cursor_color,
            (right_panel_x_start + self.cursor[0] * self.render_scale,
             self.cursor[1] * self.render_scale), 1 * self.render_scale
        )

        pygame.draw.circle(
            self.window, cursor_color,
            (self.cursor[0] * self.render_scale,
             self.cursor[1] * self.render_scale), 1 * self.render_scale
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_surface(self, array, position):
        array_rgb = np.stack([(array.T * 255)] * 3, axis=-1).astype(np.uint8)
        surface = pygame.surfarray.make_surface(array_rgb)
        scaled_surface = pygame.transform.scale(
            surface, (self.canvas_size[1] * self.render_scale, self.canvas_size[0] * self.render_scale)
        )
        self.window.blit(scaled_surface, position)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None