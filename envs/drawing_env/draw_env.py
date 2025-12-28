import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import os
import random
import pygame
from torchvision import transforms
from envs.drawing_env.tools.image_process import (
    find_starting_point,
    calculate_f1_score, calculate_metrics
)


class DrawingAgentEnv(gym.Env):
    """
    A custom Reinforcement Learning environment for a drawing agent.
    The agent learns to recreate a target sketch by moving a pen and drawing pixels.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}
    _episode_counter = 0

    def __init__(self, config=None):
        super(DrawingAgentEnv, self).__init__()
        self._init_config(config)
        self._init_state_variables()
        self.render_scale = 10

        # --- Action Space Setup ---
        # Actions 0-8: Move (3x3 grid) with Pen UP
        # Actions 9-17: Move (3x3 grid) with Pen DOWN
        # Action 18: Jump to nearest unfinished pixel (if enabled)
        n_actions = 18 + int(self.use_jump)
        self.action_space = spaces.Discrete(n_actions)

        # --- Observation Space Setup ---
        # Dynamically stack channels based on configuration
        self.num_obs_channels = 1  # Channel 0: Pen Mask (Location)

        if self.use_target_sketch_obs:
            self.num_obs_channels += 1
            print("Using Target Sketch Obs")
        if self.use_canvas_obs:
            self.num_obs_channels += 1
            print("Using Canvas Obs")
        if self.use_remaining_obs:
            self.num_obs_channels += 1
            print("Using Remaining Target Obs")

        # Shape: (Channels, Height, Width)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(self.num_obs_channels, *self.canvas_size), dtype=np.float32
        )

        self.window = None
        self.clock = None

    def _init_config(self, config):
        """Initialize configuration parameters from input dictionary or defaults."""
        config = config or {}
        self.canvas_size = config.get("canvas_size", [32, 32])
        self.max_steps = config.get("max_steps", 1000)
        self.render_mode = config.get("render_mode", None)

        # Data augmentation for target sketches
        self.aug_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2, 0.2),
                fill=255
            )
        ])

        # Action Logic Configuration
        self.use_jump = config.get("use_jump", False)
        self.jump_distance_threshold = config.get("jump_distance_threshold", 1.5)
        self.use_jump_penalty = config.get("use_jump_penalty", False)

        # Reward & Combo Logic
        self.stroke_budget = config.get("stroke_budget", 1)
        self.use_combo = config.get("use_combo", False)
        self.combo_rate = config.get("combo_rate", 1.1)
        self.repeat_scale = config.get("repeat_scale", 1.0)

        # Observation Flags
        self.use_canvas_obs = config.get("use_canvas_obs", False)
        self.use_target_sketch_obs = config.get("use_target_sketch_obs", False)
        self.use_remaining_obs = config.get("use_remaining_obs", True)

        # Reward Values
        self.reward_correct = config.get("reward_correct", 0.1)
        self.reward_wrong = config.get("reward_wrong", -0.01)
        self.reward_jump = config.get("reward_jump", 0.0)
        self.penalty_scale_threshold = config.get("penalty_scale_threshold", 0.9)
        self.jump_penalty = config.get("jump_penalty", -0.5)

        # Tool Config
        self.brush_size = config.get("brush_size", 1)
        self.use_augmentation = config.get("use_augmentation", True)

        # Dataset Loading
        self.target_sketches_path = config.get("target_sketches_path", None)
        self.specific_sketch_file = config.get("specific_sketch_file", None)
        self.sketch_file_list = []

        # Determine if loading a specific file or a directory
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

        self.current_episode_num = 0

    def _init_state_variables(self):
        """Reset internal state variables for a new episode."""
        self.current_step = 0
        # Canvas initialized to White (1.0)
        self.canvas = np.full(self.canvas_size, 1.0, dtype=np.float32)
        self.used_budgets = 0
        self.is_pen_down = False
        self.pen_was_down = False
        self.episode_end = False

        # Combo tracking
        self.current_combo = 0
        self.combo_sustained_on_repeat = 0
        self.episode_combo_log = []
        self.episode_combo_sustained_on_repeat_log = []

        # Metrics
        self.target_pixel_count = 0
        self._current_tp = 0
        self._current_tn = 0
        self._current_fp = 0
        self._current_fn = 0

        self.last_pixel_similarity = 0.0
        self.last_recall_black = 0.0
        self.last_recall_white = 0.0
        self.last_precision_black = 0.0
        self.last_f1_score = 0.0

        # Episode Accumulators
        self.episode_return = 0
        self.target_sketch = None
        self.cursor = [0, 0]
        self.episode_total_painted = 0
        self.episode_correctly_painted = 0
        self.episode_base_reward = 0.0
        self.episode_combo_bonus = 0.0
        self.episode_negative_reward = 0.0
        self.episode_jump_count = 0
        self.painted_pixels_since_last_jump = 0

        self.episode_jump_draw_combo_count = 0
        self.last_raw_action = None

    def _decode_action(self, action):
        """
        Decodes discrete action into semantic components.
        Returns: (dx, dy, is_pen_down, is_jump)
        """
        # 0-8:  Move (dx, dy) + Pen Up
        # 9-17: Move (dx, dy) + Pen Down
        # 18:   Jump (Only if use_jump=True)
        is_jump = False
        dx, dy = 0, 0
        is_pen_down = 0
        if self.use_jump and action == 18:
            is_jump = True
        else:
            is_pen_down = (action >= 9)
            sub_action = action % 9
            # Map 0-8 to (-1, -1) through (1, 1)
            dx = (sub_action % 3) - 1
            dy = (sub_action // 3) - 1

        return dx, dy, is_pen_down, is_jump

    def _load_sketch_from_path(self, filepath):
        """Loads, converts to grayscale, augments, and normalizes the target image."""
        sketch = Image.open(filepath).convert('L')
        if self.use_augmentation:
            sketch = self.aug_transform(sketch)
        sketch_array = np.array(sketch)
        return (sketch_array / 255.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment to start a new episode."""
        super().reset(seed=seed)
        DrawingAgentEnv._episode_counter += 1
        self.current_episode_num = DrawingAgentEnv._episode_counter
        self._init_state_variables()
        self.correct_rewards = 0

        # Select and load a random target sketch
        if not self.sketch_file_list:
            raise ValueError("Sketch file list is empty!")
        chosen_file = random.choice(self.sketch_file_list)
        self.target_sketch = self._load_sketch_from_path(chosen_file)

        # Calculate initial target stats (Black < 0.5)
        self.target_pixel_count = np.sum(self.target_sketch < 0.5)
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
        """Finds the closest unpainted 'black' pixel in the target sketch."""
        unfinished = np.logical_and(
            self.target_sketch < 0.5,
            self.canvas > 0.5
        )

        unfinished_indices = np.argwhere(unfinished)

        if len(unfinished_indices) == 0:
            return self.cursor, 0.0

        current_r, current_c = self.cursor[1], self.cursor[0]

        dists = np.linalg.norm(
            unfinished_indices - np.array([current_r, current_c]),
            axis=1
        )

        min_idx = np.argmin(dists)
        target_r, target_c = unfinished_indices[min_idx]

        return [target_c, target_r], dists[min_idx]

    def step(self, action):
        """
        Executes one time step within the environment.
        Handles movement, painting, reward calculation, and metric updates.
        """
        self.current_step += 1

        ACTION_JUMP = 18
        ACTION_DRAW_IN_PLACE = 13

        current_action = int(action)

        # Track "Jump -> Draw" combo pattern
        if self.use_jump and self.last_raw_action is not None:
            if self.last_raw_action == ACTION_JUMP and current_action == ACTION_DRAW_IN_PLACE:
                self.episode_jump_draw_combo_count += 1

        self.last_raw_action = current_action

        dx, dy, is_pen_down, is_jump = self._decode_action(action)

        # --- JUMP LOGIC ---
        jump_penalty = 0.0
        if self.use_jump and is_jump:
            nearest_pos, dist = self._find_nearest_target_pixel()
            # Apply penalty if jump is too short (discourage useless jumps)
            if self.use_jump_penalty:
                if dist <= self.jump_distance_threshold:
                    jump_penalty = self.jump_penalty
                else:
                    jump_penalty = self.reward_jump

            # Teleport cursor
            self.cursor[0] = np.clip(nearest_pos[0], 0, self.canvas_size[0] - 1)
            self.cursor[1] = np.clip(nearest_pos[1], 0, self.canvas_size[1] - 1)

            self.episode_jump_count += 1
            self.painted_pixels_since_last_jump = 0
        else:
            # Standard Move
            self._update_agent_state(dx, dy, bool(is_pen_down))

        terminated = False
        correct_new_pixels = []
        repeated_correct_pixels = []

        # --- PAINTING LOGIC ---
        if is_pen_down and not terminated:
            valid_paint_count = 0
            current_cursor = self.cursor
            brush_radius = self.brush_size // 2

            # Define brush area
            y_start = max(0, current_cursor[1] - brush_radius)
            y_end = min(self.canvas_size[1], current_cursor[1] + brush_radius + 1)
            x_start = max(0, current_cursor[0] - brush_radius)
            x_end = min(self.canvas_size[0], current_cursor[0] + brush_radius + 1)

            # Analyze pixels under brush
            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):  # Canvas is currently white
                        if np.isclose(self.target_sketch[r, c], 0.0):  # Target is black (Correct)
                            correct_new_pixels.append((r, c))
                            valid_paint_count += 1
                            self._current_tp += 1
                            self._current_fn -= 1
                        else:  # Target is white (Wrong)
                            self._current_fp += 1
                            self._current_tn -= 1
                    else:  # Pixel already painted
                        if np.isclose(self.target_sketch[r, c], 0.0):
                            repeated_correct_pixels.append((r, c))

            self.painted_pixels_since_last_jump += valid_paint_count

            # Apply paint to canvas (Set to 0.0/Black)
            for r in range(y_start, y_end):
                for c in range(x_start, x_end):
                    if np.isclose(self.canvas[r, c], 1.0):
                        self.episode_total_painted += 1
                        self.canvas[r, c] = 0.0

        # Update metrics
        current_recall_black, current_recall_white, current_precision_black, current_pixel_similarity = calculate_metrics(
            self._current_tp, self._current_fp, self._current_tn, self._current_fn, self.canvas.size
        )

        self.last_pixel_similarity = current_pixel_similarity
        self.last_recall_black = current_recall_black
        self.last_recall_white = current_recall_white
        self.last_precision_black = current_precision_black

        current_f1_score = calculate_f1_score(self.last_precision_black, self.last_recall_black)

        # Check termination conditions
        truncated = self.current_step >= self.max_steps or np.isclose(self.last_recall_black, 1.0)

        # Calculate Reward
        reward = self._calculate_reward(
            is_pen_down,
            correct_new_pixels,
            repeated_correct_pixels,
        )

        # Add jump penalty if applicable
        if is_jump:
            reward += jump_penalty
            if jump_penalty < 0:
                self.episode_negative_reward += jump_penalty

        self.last_f1_score = current_f1_score

        # Finalize episode logs
        if terminated or truncated:
            if self.current_combo > 0:
                self.episode_combo_log.append(self.current_combo)
            if self.combo_sustained_on_repeat > 0:
                self.episode_combo_sustained_on_repeat_log.append(self.combo_sustained_on_repeat)
            self.episode_end = True

        observation = self._get_obs()
        info = self._get_info()
        self.episode_return += reward

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _update_agent_state(self, dx, dy, is_pen_down):
        """Updates cursor position and pen state."""
        self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.canvas_size[0] - 1)
        self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.canvas_size[1] - 1)

        # Count budget usage (pen strokes)
        if self.pen_was_down and not is_pen_down:
            self.used_budgets = min(255, self.used_budgets + 1)

        self.is_pen_down = is_pen_down
        self.pen_was_down = self.is_pen_down

    def _calculate_reward(self, is_pen_down, correct_new_pixels, repeated_new_pixels):
        """
        Calculates reward based on drawing accuracy and combo mechanics.
        Rewards continuous correct strokes; penalizes mistakes and redundant painting.
        """
        reward = 0.0
        drawing_reward = 0.0
        base_reward_part = 0.0
        bonus_reward_part = 0.0

        if is_pen_down:
            num_correct = len(correct_new_pixels)
            num_repeated = len(repeated_new_pixels)
            self.episode_correctly_painted += num_correct

            if num_correct > 0:
                # --- Reward for New Correct Pixels ---
                positive_reward_this_step = num_correct * self.reward_correct

                base_reward_part = positive_reward_this_step
                self.current_combo += 1
                self.combo_sustained_on_repeat += 1

                # Apply Combo Multiplier
                if self.use_combo:
                    if self.combo_rate < 1.0:
                        positive_reward_this_step *= (1 + self.combo_rate * self.combo_sustained_on_repeat)
                    else:
                        positive_reward_this_step *= (self.combo_rate ** self.combo_sustained_on_repeat)

                bonus_reward_part = positive_reward_this_step - base_reward_part
                drawing_reward = positive_reward_this_step
                self.correct_rewards += positive_reward_this_step

            elif num_correct == 0 and num_repeated > 0:
                # --- Penalty for Repeating (Redrawing already black pixels) ---
                if self.current_combo > 0:
                    self.episode_combo_log.append(self.current_combo)
                self.current_combo = 0
                self.combo_sustained_on_repeat += 1  # Sustain the underlying stroke count

                # Scale penalty based on progress (threshold)
                current_penalty_scale = 0.0
                if self.penalty_scale_threshold > 0:
                    if self.penalty_scale_threshold <= self.last_recall_black < 1.0:
                        current_penalty_scale = self.last_recall_black
                    elif self.last_recall_black >= 1.0 or self.penalty_scale_threshold > 1.0:
                        current_penalty_scale = 1.0

                negative_reward_this_step = self.reward_wrong * self.repeat_scale * current_penalty_scale
                self.episode_negative_reward += negative_reward_this_step
                drawing_reward = negative_reward_this_step

            elif num_correct == 0 and num_repeated == 0:
                # --- Penalty for Wrong Pixels (Drawing on White) ---
                if self.current_combo > 0:
                    self.episode_combo_log.append(self.current_combo)
                if self.combo_sustained_on_repeat > 0:
                    self.episode_combo_sustained_on_repeat_log.append(self.combo_sustained_on_repeat)

                # Reset combos
                self.current_combo = 0
                self.combo_sustained_on_repeat = 0

                current_penalty_scale = 0.0
                if self.penalty_scale_threshold > 0:
                    if self.penalty_scale_threshold <= self.last_recall_black <= 1.0:
                        current_penalty_scale = self.last_recall_black
                    elif self.penalty_scale_threshold > 1.0:
                        current_penalty_scale = 1.0

                negative_reward_this_step = self.reward_wrong * current_penalty_scale
                self.episode_negative_reward += negative_reward_this_step
                drawing_reward = negative_reward_this_step
        else:
            # --- Pen Up: Reset Combos ---
            if self.current_combo > 0:
                self.episode_combo_log.append(self.current_combo)
            if self.combo_sustained_on_repeat > 0:
                self.episode_combo_sustained_on_repeat_log.append(self.combo_sustained_on_repeat)
            self.current_combo = 0
            self.combo_sustained_on_repeat = 0

        reward += drawing_reward
        self.episode_base_reward += base_reward_part
        self.episode_combo_bonus += bonus_reward_part
        return reward

    def _get_obs(self):
        """
        Constructs the observation tensor.
        Stack: [Target Sketch, Canvas, Difference, Pen Mask] (depending on config)
        """
        if not hasattr(self, "_obs_img"):
            self._obs_img = np.zeros((self.num_obs_channels, *self.canvas_size), dtype=np.float32)

        ch_idx = 0

        # Channel: Target Sketch
        if self.use_target_sketch_obs:
            if self.current_step == 0:
                self._obs_img[ch_idx] = self.target_sketch.astype(np.float32)
            self._obs_img[ch_idx] = self.target_sketch
            ch_idx += 1

        # Channel: Current Canvas
        if self.use_canvas_obs:
            self._obs_img[ch_idx][:] = self.canvas
            ch_idx += 1

        # Channel: Remaining Target (Difference)
        if self.use_remaining_obs:
            diff_map = self.canvas - self.target_sketch
            diff_map = np.clip(diff_map, 0.0, 1.0)
            self._obs_img[ch_idx][:] = diff_map
            ch_idx += 1

        # Channel: Pen Position Mask (Always included)
        pen_layer = self._obs_img[ch_idx]
        pen_layer.fill(0.0)

        y, x = self.cursor[1], self.cursor[0]
        if 0 <= y < self.canvas_size[0] and 0 <= x < self.canvas_size[1]:
            pen_layer[y, x] = 1.0
        ch_idx += 1

        return self._obs_img

    def _get_info(self):
        """Returns dictionary of metrics for logging/debugging."""
        info_dict = {
            "pixel_similarity": self.last_pixel_similarity,
            "recall_black": self.last_recall_black,
            "recall_white": self.last_recall_white,
            "used_budgets": self.used_budgets,
            "episode_return": self.episode_return,
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
            "jump_count": self.episode_jump_count,
            "jump_draw_combo_count": self.episode_jump_draw_combo_count,
            "target_pixel_count": self.target_pixel_count,
        }
        return info_dict

    def _init_pygame(self):
        """Initializes the Pygame window for rendering."""
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
        """Renders the current environment state (Target vs Canvas) using Pygame."""
        if self.render_mode != "human": return
        if self.window is None or self.clock is None: self._init_pygame()

        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise Exception("Pygame window closed by user.")

        self.window.fill((105, 105, 105))  # Grey background

        # Draw Target Sketch (Left)
        self._draw_surface(self.target_sketch, (0, 0))

        # Draw Canvas (Right)
        right_panel_x_start = (self.canvas_size[1] + 10) * self.render_scale
        self._draw_surface(self.canvas, (right_panel_x_start, 0))

        # Draw Cursor on both panels
        cursor_color = (255, 0, 0) if self.is_pen_down else (0, 0, 255)  # Red=Down, Blue=Up

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
        """Helper to draw a numpy array onto the Pygame surface."""
        array_rgb = np.stack([(array.T * 255)] * 3, axis=-1).astype(np.uint8)
        surface = pygame.surfarray.make_surface(array_rgb)
        scaled_surface = pygame.transform.scale(
            surface, (self.canvas_size[1] * self.render_scale, self.canvas_size[0] * self.render_scale)
        )
        self.window.blit(scaled_surface, position)

    def close(self):
        """Closes the rendering window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None