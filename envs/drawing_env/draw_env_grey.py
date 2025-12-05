import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import os
import pygame
from envs.drawing_env.tools.image_process import *
from collections import deque


def _decode_action(action):
    is_pen_down = action >= 9
    sub_action = action % 9
    dx = (sub_action % 3) - 1
    dy = (sub_action // 3) - 1
    return dx, dy, int(is_pen_down), 0 # disable stop action


class DrawingAgentGreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    _episode_counter = 0

    def __init__(self, config=None):
        super(DrawingAgentGreyEnv, self).__init__()
        self._init_config(config)
        self._init_state_variables()
        self.render_scale = 10

        self.action_space = spaces.Discrete(18)

        num_obs_channels = (
                1 +  # Pen Mask (Always included in logic below)
                int(self.use_budget_channel) +
                int(self.use_canvas_obs) +
                int(self.use_target_sketch_obs) +
                int(self.use_difference_map_obs) +
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
        self.combo_rate = config.get("combo_rate", 0.1)
        self.max_combo_normalization = 50.0

        # Obs Flags
        self.use_canvas_obs = config.get("use_canvas_obs", True)
        self.use_target_sketch_obs = config.get("use_target_sketch_obs", True)
        self.use_stroke_trajectory_obs = config.get("use_stroke_trajectory_obs", False)
        self.use_difference_map_obs = config.get("use_difference_map_obs", True)

        # Penalties & Rewards
        self.reward_correct = config.get("reward_correct", 0.1)
        self.reward_wrong = config.get("reward_wrong", -0.01)
        self.use_time_penalty = config.get("use_time_penalty", False)
        self.use_mvg_penalty_compensation = config.get("use_mvg_penalty_compensation", False)
        self.mvg_penalty_window_size = self.max_steps // 5
        self.penalty_scale_threshold = config.get("penalty_scale_threshold", 0.9)

        self.similarity_weight = config.get("similarity_weight", 0.0)
        self.recall_bonus = config.get("recall_bonus", 0.0)
        self.f1_scalar = config.get("f1_scalar", 0.0)

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
        self.last_recall_all = 0.0
        self.last_recall_white = 0.0
        self.last_recall_grey = 0.0
        self.last_precision = 0.0
        self.last_f1_score = 0.0

        self.step_rewards = 0
        self.target_sketch = None
        self.cursor = [0, 0]
        self.episode_total_painted = 0
        self.episode_correctly_painted = 0

        self.penalty_history = deque(maxlen=self.mvg_penalty_window_size)
        self.current_mvg_penalty = 0.0
        self.current_episode_step_data = []

        self.episode_base_reward = 0.0
        self.episode_combo_bonus = 0.0

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
        DrawingAgentGreyEnv._episode_counter += 1
        self.current_episode_num = DrawingAgentGreyEnv._episode_counter
        self._init_state_variables()
        self.correct_rewards = 0

        # Load Target
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
        else:
            if not self.sketch_file_list:
                raise ValueError("Sketch file list is empty!")
            chosen_file = random.choice(self.sketch_file_list)
            self.target_sketch = self._load_sketch_from_path(chosen_file)
        self.cursor = find_starting_point(self.target_sketch)

        # Initialize Metrics based on Target vs Empty Canvas (1.0)
        # At start, Canvas is all White (1.0)
        target_is_white = np.isclose(self.target_sketch, 1.0)
        target_not_white = ~target_is_white

        # FN: Target has ink (0.0 or 0.5), Canvas is White (1.0) -> Mismatch
        self._current_fn = np.sum(target_not_white)
        # TN: Target is White, Canvas is White -> Match
        self._current_tn = np.sum(target_is_white)
        # TP: Canvas has ink match? No, canvas is empty.
        self._current_tp = 0
        # FP: Canvas has ink wrong? No, canvas is empty.
        self._current_fp = 0

        self.last_recall_white = 1.0 if self._current_tn > 0 else 0.0

        if self.render_mode == "human":
            self._init_pygame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        dx, dy, is_pen_down, is_stop_action = _decode_action(action)

        self._update_agent_state(dx, dy, bool(is_pen_down), is_stop_action)
        terminated = is_stop_action

        reward_info = {
            'painted_pixels': [],  # (r, c, old_val, new_val)
            'attempted_paint': [],  # (r, c) where pen was down but no value change (e.g. 0.0 -> 0.0)
            'wrong_on_white': False
        }

        if is_pen_down and not terminated:
            current_cursor = self.cursor
            brush_radius = self.brush_size // 2

            # Calculate affected area
            y_start = max(0, current_cursor[1] - brush_radius)
            y_end = min(self.canvas_size[1], current_cursor[1] + brush_radius + 1)
            x_start = max(0, current_cursor[0] - brush_radius)
            x_end = min(self.canvas_size[0], current_cursor[0] + brush_radius + 1)

            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    old_val = self.canvas[r, c]
                    new_val = max(0.0, old_val - 0.5)

                    if not np.isclose(old_val, new_val):
                        self.canvas[r, c] = new_val
                        self.episode_total_painted += 1
                        reward_info['painted_pixels'].append((r, c, old_val, new_val))
                    else:
                        reward_info['attempted_paint'].append((r, c))

        current_recall_black, current_recall_grey, current_recall_all, current_recall_white, current_precision, current_pixel_similarity = calculate_metrics_grey(
            self.target_sketch, self.canvas
        )

        self.last_pixel_similarity = current_pixel_similarity
        self.last_recall_grey = current_recall_grey
        self.last_recall_black = current_recall_black
        self.last_recall_white = current_recall_white
        self.last_recall_all = current_recall_all
        self.last_precision = current_precision

        current_f1_score = calculate_f1_score(self.last_precision, self.last_recall_black)

        truncated = self.current_step >= self.max_steps or (
                    self.last_recall_black >= 0.99 and self.last_recall_grey >= 0.99)

        reward = self._calculate_reward(
            is_pen_down,
            reward_info,
        )
        self.last_f1_score = current_f1_score

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

    def _calculate_reward(self, is_pen_down, reward_info):
        reward = 0.0 if not self.use_time_penalty else -0.001

        # Specific Reward Values requested
        R_GOOD = self.reward_correct
        R_BAD_DRAW = self.reward_wrong

        base_reward = 0.0
        bonus_reward = 0.0
        drawing_reward = 0.0

        painted_pixels = reward_info['painted_pixels']
        attempted_paint = reward_info['attempted_paint']

        any_repeat_correct_paint = False
        any_bad_paint = False

        if is_pen_down:
            # Check Pixel Changes
            for (r, c, old_val, new_val) in painted_pixels:
                target_val = self.target_sketch[r, c]
                current_reward = 0.0
                # Condition 1: Good Paint (Approaching target, not overshooting)
                if new_val >= target_val:
                    # e.g. T=0.0, Old=0.5, New=0.0 -> OK (0>=0)
                    # e.g. T=0.5, Old=1.0, New=0.5 -> OK (0.5>=0.5)
                    current_reward = R_GOOD
                    self.episode_correctly_painted += 1

                # Condition 2: Overshoot (Grey -> Black) or (White -> Grey)
                else:
                    current_reward = R_BAD_DRAW
                    any_bad_paint = True

                # Apply Combo to POSITIVE rewards only
                if current_reward > 0:
                    base_reward += current_reward
                    self.current_combo += 1
                    self.combo_sustained_on_repeat += 1
                    # Calculate Bonus
                    multiplier = (self.combo_rate ** self.combo_sustained_on_repeat) if self.combo_rate >= 1.0 else (
                                1 + self.combo_rate * self.combo_sustained_on_repeat)
                    total_pixel_reward = current_reward * multiplier
                    bonus_reward += (total_pixel_reward - current_reward)
                    drawing_reward += total_pixel_reward
                else:
                    current_penalty_scale = 0.0
                    if self.penalty_scale_threshold > 0:
                        if self.penalty_scale_threshold <= self.last_recall_all < 1.0:
                            current_penalty_scale = self.last_precision
                        elif self.last_recall_all >= 1.0 or self.penalty_scale_threshold > 1.0:
                            current_penalty_scale = 1.0
                    drawing_reward += current_reward * current_penalty_scale

            # Check No-Change Paints (Repeats)
            for (r, c) in attempted_paint:
                target_val = self.target_sketch[r, c]
                old_val = self.canvas[r, c]
                # Condition: Repeated Black
                if np.isclose(target_val, 0.0) and np.isclose(old_val, 0.0):
                    drawing_reward += R_BAD_DRAW
                    any_repeat_correct_paint = True

            if any_bad_paint or any_repeat_correct_paint:
                if self.current_combo > 0: self.episode_combo_log.append(self.current_combo)
                self.current_combo = 0

            if any_bad_paint:
                if self.combo_sustained_on_repeat > 0: self.episode_combo_sustained_on_repeat_log.append(
                    self.combo_sustained_on_repeat)
                self.combo_sustained_on_repeat = 0

        else:
            # Pen Up Logic
            if self.current_combo > 0: self.episode_combo_log.append(self.current_combo)
            if self.combo_sustained_on_repeat > 0: self.episode_combo_sustained_on_repeat_log.append(
                self.combo_sustained_on_repeat)
            self.current_combo = 0
            self.combo_sustained_on_repeat = 0

        reward += drawing_reward
        self.episode_base_reward += base_reward
        self.episode_combo_bonus += bonus_reward
        self.step_rewards += reward
        return reward

    def _get_info(self):
        info_dict = {
            "pixel_similarity": self.last_pixel_similarity,
            "recall_black": self.last_recall_black,
            "recall_grey": self.last_recall_grey,
            "recall_all": self.last_recall_all,
            "recall_white": self.last_recall_white,
            "used_budgets": self.used_budgets,
            "step_rewards": self.step_rewards,
            "total_painted": self.episode_total_painted,
            "correctly_painted": self.episode_correctly_painted,
            "combo_count": self.current_combo,
            "precision": self.last_precision,
            "f1_score": self.last_f1_score,
            "episode_combo_log": self.episode_combo_log,
            "episode_base_reward": self.episode_base_reward,
            "episode_combo_bonus": self.episode_combo_bonus,
            "combo_sustained": self.episode_combo_sustained_on_repeat_log,
        }
        return info_dict

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

        # 3. Difference Map
        if self.use_difference_map_obs:
            diff_map = np.clip(self.canvas - self.target_sketch, 0.0, 1.0)
            self._obs[ch_idx][:] = diff_map
            ch_idx += 1

        # 4. Pen Mask
        pen_mask = self._obs[ch_idx]
        pen_mask.fill(0.0)
        y, x = self.cursor[1], self.cursor[0]
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            pen_mask[y, x] = 1.0
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