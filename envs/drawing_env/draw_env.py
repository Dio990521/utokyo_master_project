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


class DrawingAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    _episode_counter = 0

    def __init__(self, config=None):
        super(DrawingAgentEnv, self).__init__()
        self._init_config(config)
        self._init_state_variables()
        self.render_scale = 10

        # Action Space Setup
        # simplified=True:  10 actions (0-8: Draw+Move, 9: Jump)
        # simplified=False: 18 actions (0-8: Move, 9-17: Draw+Move) [+1 if Jump enabled]
        if self.use_simplified_action_space:
            self.action_space = spaces.Discrete(10)
        else:
            n_actions = 18
            if self.use_jump:
                n_actions += 1  # Action 18 is Jump
            self.action_space = spaces.Discrete(n_actions)

        self.num_obs_channels = 1  # Pen Mask is always included in logic

        if self.use_canvas_obs:
            self.num_obs_channels += 1
        if self.use_target_sketch_obs:
            self.num_obs_channels += 1
        if self.use_stroke_trajectory_obs:
            self.num_obs_channels += 1
        img_space = spaces.Box(
            low=0, high=1.0, shape=(self.num_obs_channels, *self.canvas_size), dtype=np.float32
        )
        if self.use_dist_val_obs:
            self.observation_space = spaces.Dict({
                "image": img_space,
                "jump_counter": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })
        else:
            self.observation_space = img_space

        self.window = None
        self.clock = None

    def _init_config(self, config):
        config = config or {}
        self.canvas_size = config.get("canvas_size", [32, 32])
        self.max_steps = config.get("max_steps", 1000)
        self.render_mode = config.get("render_mode", None)

        # Action Space Configuration
        self.use_simplified_action_space = config.get("use_simplified_action_space", False)
        self.use_jump = config.get("use_jump", False)
        self.use_jump_penalty =config.get("use_jump_penalty", False)
        self.use_dist_val_obs = config.get("use_jump_counter_obs", self.use_jump_penalty)

        # Budget & Combo (Logic retained for rewards, channels removed)
        self.stroke_budget = config.get("stroke_budget", 1)
        self.use_combo = config.get("use_combo", False)
        self.combo_rate = config.get("combo_rate", 1.1)

        # Obs Flags
        self.use_canvas_obs = config.get("use_canvas_obs", True)
        self.use_target_sketch_obs = config.get("use_target_sketch_obs", True)
        self.use_stroke_trajectory_obs = config.get("use_stroke_trajectory_obs", False)

        # Penalties & Rewards
        self.reward_correct = config.get("reward_correct", 0.1)
        self.reward_wrong = config.get("reward_wrong", -0.01)
        self.use_time_penalty = config.get("use_time_penalty", False)
        self.use_mvg_penalty_compensation = config.get("use_mvg_penalty_compensation", False)
        self.mvg_penalty_window_size = self.max_steps // 5
        self.penalty_scale_threshold = config.get("penalty_scale_threshold", 0.9)

        self.use_skeleton_guidance = config.get("use_skeleton_guidance", False)

        # Drawing Config
        self.brush_size = config.get("brush_size", 1)

        # Data Configs
        self.target_sketches_path = config.get("target_sketches_path", None)
        self.specific_sketch_file = config.get("specific_sketch_file", None)
        self.sketch_file_list = []
        if self.specific_sketch_file and os.path.exists(self.specific_sketch_file):
            self.sketch_file_list = [self.specific_sketch_file]
        elif self.target_sketches_path and os.path.exists(self.target_sketches_path):
            self.sketch_file_list = [
                os.path.join(self.target_sketches_path, f)
                for f in os.listdir(self.target_sketches_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not self.sketch_file_list:
                raise ValueError(f"No image files found in {self.target_sketches_path}")

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

        self.episode_base_reward = 0.0
        self.episode_combo_bonus = 0.0
        self.episode_negative_reward = 0.0
        self.episode_jump_count = 0
        self.painted_pixels_since_last_jump = 0

    def _decode_action(self, action):
        is_jump = False
        dx, dy = 0, 0
        is_pen_down = 0

        if self.use_simplified_action_space:
            # 0-8: Move (dx, dy) + Pen Down (Always Drawing)
            # 9:   Jump
            if action == 9:
                is_jump = True
            else:
                is_pen_down = 1
                dx = (action % 3) - 1
                dy = (action // 3) - 1
        else:
            # 0-8:  Move (dx, dy) + Pen Up
            # 9-17: Move (dx, dy) + Pen Down
            # 18:   Jump (Only if use_jump=True)
            if self.use_jump and action == 18:
                is_jump = True
            else:
                is_pen_down = (action >= 9)
                sub_action = action % 9
                dx = (sub_action % 3) - 1
                dy = (sub_action // 3) - 1

        return dx, dy, is_pen_down, 0, is_jump

    def _load_sketch_from_path(self, filepath):
        sketch = Image.open(filepath).convert('L')
        sketch_array = np.array(sketch)
        return (sketch_array / 255.0).astype(np.float32)

    def _jump_to_random_endpoint(self):
        endpoints = get_active_endpoints(self.target_sketch, self.canvas)
        if endpoints is not None and len(endpoints[0]) > 0:
            num_points = len(endpoints[0])
            idx = np.random.randint(0, num_points)
            y = endpoints[0][idx]
            x = endpoints[1][idx]
            self.cursor[0] = x
            self.cursor[1] = y
        else:
            # Fallback to random ink location
            self._jump_to_random_ink_location()

    def _jump_to_random_ink_location(self):
        unfinished_mask = (self.target_sketch < 0.5) & (self.canvas > 0.5)
        target_indices = np.argwhere(unfinished_mask)
        if len(target_indices) > 0:
            idx = np.random.randint(0, len(target_indices))
            y, x = target_indices[idx]
            self.cursor[0] = x
            self.cursor[1] = y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        DrawingAgentEnv._episode_counter += 1
        self.current_episode_num = DrawingAgentEnv._episode_counter
        self._init_state_variables()
        self.correct_rewards = 0

        # Load Target
        if not self.sketch_file_list:
            raise ValueError("Sketch file list is empty!")
        chosen_file = random.choice(self.sketch_file_list)
        self.target_sketch = self._load_sketch_from_path(chosen_file)

        self.cursor = find_starting_point(self.target_sketch)

        target_is_black = np.isclose(self.target_sketch, 0.0)
        target_is_white = np.isclose(self.target_sketch, 1.0)
        self._current_tp = 0
        self._current_fp = 0
        self._current_tn = np.sum(target_is_white)
        self._current_fn = np.sum(target_is_black)

        self.last_recall_white = 1.0 if self._current_tn > 0 else 0.0

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), self._get_info()

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
        dx, dy, is_pen_down, _, is_jump = self._decode_action(action)

        jump_penalty = 0.0
        if is_jump:
            _, dist = self._find_nearest_target_pixel()

            if dist <= 1.0:
                jump_penalty = -2.0
            else:
                jump_penalty = 0.1

            self._jump_to_random_endpoint()
            self.episode_jump_count += 1
            self.painted_pixels_since_last_jump = 0

        terminated = False
        self._update_agent_state(dx, dy, bool(is_pen_down), False)

        correct_new_pixels = []
        repeated_correct_pixels = []

        if is_pen_down and not terminated:
            valid_paint_count = 0
            current_cursor = self.cursor
            brush_radius = self.brush_size // 2

            y_start = max(0, current_cursor[1] - brush_radius)
            y_end = min(self.canvas_size[1], current_cursor[1] + brush_radius + 1)
            x_start = max(0, current_cursor[0] - brush_radius)
            x_end = min(self.canvas_size[0], current_cursor[0] + brush_radius + 1)

            # Check pixels
            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):
                        if np.isclose(self.target_sketch[r, c], 0.0):
                            correct_new_pixels.append((r, c))
                            valid_paint_count += 1
                            self._current_tp += 1
                            self._current_fn -= 1
                        else:
                            self._current_fp += 1
                            self._current_tn -= 1
                    else:
                        if np.isclose(self.target_sketch[r, c], 0.0):
                            repeated_correct_pixels.append((r, c))

            self.painted_pixels_since_last_jump += valid_paint_count

            # Apply paint
            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):
                        self.episode_total_painted += 1
                        self.canvas[r, c] = 0.0

        current_recall_black, current_recall_white, current_precision_black, current_pixel_similarity = calculate_metrics(
            self._current_tp, self._current_fp, self._current_tn, self._current_fn, self.canvas.size
        )

        self.last_pixel_similarity = current_pixel_similarity
        self.last_recall_black = current_recall_black
        self.last_recall_white = current_recall_white
        self.last_precision_black = current_precision_black

        current_f1_score = calculate_f1_score(self.last_precision_black, self.last_recall_black)

        truncated = self.current_step >= self.max_steps or np.isclose(self.last_recall_black, 1.0)

        reward = self._calculate_reward(
            is_pen_down,
            correct_new_pixels,
            repeated_correct_pixels,
        )

        if is_jump:
            reward += jump_penalty
            if jump_penalty < 0:
                self.episode_negative_reward += jump_penalty

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

    def _calculate_reward(self, is_pen_down, correct_new_pixels, repeated_new_pixels):
        reward = 0.0 if not self.use_time_penalty else -0.001
        drawing_reward = 0.0
        base_reward_part = 0.0
        bonus_reward_part = 0.0

        if is_pen_down:
            num_correct = len(correct_new_pixels)
            num_repeated = len(repeated_new_pixels)
            self.episode_correctly_painted += num_correct

            if num_correct > 0:
                positive_reward_this_step = num_correct * self.reward_correct

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
                        current_penalty_scale = self.last_recall_black
                    elif self.last_recall_black >= 1.0 or self.penalty_scale_threshold > 1.0:
                        current_penalty_scale = 1.0

                negative_reward_this_step = self.reward_wrong * 0.5 * current_penalty_scale
                self.episode_negative_reward += negative_reward_this_step
                drawing_reward = negative_reward_this_step
            elif num_correct == 0 and num_repeated == 0:
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

        reward += drawing_reward
        self.episode_base_reward += base_reward_part
        self.episode_combo_bonus += bonus_reward_part
        self.step_rewards += reward
        return reward

    def _get_obs(self):
        if not hasattr(self, "_obs_img"):
            self._obs_img = np.zeros((self.num_obs_channels, *self.canvas_size), dtype=np.float32)

        ch_idx = 0

        # Channel 1: Canvas
        if self.use_canvas_obs:
            self._obs_img[ch_idx][:] = self.canvas
            ch_idx += 1

        # Channel 2: Target
        if self.use_target_sketch_obs:
            if self.current_step == 0:
                self._obs_img[ch_idx] = self.target_sketch.astype(np.float32)
            self._obs_img[ch_idx] = self.target_sketch
            ch_idx += 1

        # Channel 3: Pen Mask (Always included)
        pen_layer = self._obs_img[ch_idx]
        pen_layer.fill(0.0)

        if self.use_skeleton_guidance:
            active_endpoints = get_active_endpoints(self.target_sketch, self.canvas)
            if active_endpoints is not None:
                pen_layer[active_endpoints] = 0.5

        y, x = self.cursor[1], self.cursor[0]
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            pen_layer[y, x] = 1.0
        ch_idx += 1

        # Channel 4: Trajectory
        if self.use_stroke_trajectory_obs:
            traj_map = self._obs_img[ch_idx]
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

        if self.use_dist_val_obs:
            # image + scalar vector
            _, dist = self._find_nearest_target_pixel()
            max_dist = np.linalg.norm(self.canvas_size)
            dist_val = min(dist / max_dist, 1.0)
            return {
                "image": self._obs_img,
                "dist_val": np.array([dist_val], dtype=np.float32)
            }

        return self._obs_img

    def _get_info(self):
        info_dict = {
            "pixel_similarity": self.last_pixel_similarity,
            "recall_black": self.last_recall_black,
            "recall_white": self.last_recall_white,
            "used_budgets": self.used_budgets,
            "step_rewards": self.step_rewards,
            "total_painted": self.episode_total_painted,
            "correctly_painted": self.episode_correctly_painted,
            "combo_count": self.current_combo,
            "precision": self.last_precision_black,
            "f1_score": self.last_f1_score,
            "episode_combo_log": self.episode_combo_log,
            "episode_base_reward": self.episode_base_reward,
            "episode_combo_bonus": self.episode_combo_bonus,
            "combo_sustained": self.episode_combo_sustained_on_repeat_log,
            "negative_reward": self.episode_negative_reward,
            "jump_count": self.episode_jump_count
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