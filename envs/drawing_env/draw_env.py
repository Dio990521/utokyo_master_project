import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import os
import random
import pygame

from envs.drawing_env.tools.image_process import calculate_similarity, draw_line_on_canvas, \
    calculate_shape_similarity_distance, calculate_pixel_similarity


class DrawingAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}  # 提高渲染帧率

    def __init__(self, target_sketches_path="sketches/", config=None):
        super(DrawingAgentEnv, self).__init__()
        self.last_pixel_similarity = None
        self.canvas_size = config.get("canvas_size", [100, 100])
        self.max_steps = config.get("max_steps", 1000)
        self.render_mode = config.get("render_mode", "human")
        self.current_step = 0
        self.max_hp = config.get("max_hp", 100)
        self.hp = self.max_hp
        self.target_sketches_path = target_sketches_path
        self.target_sketches = self._load_target_sketches()
        if not self.target_sketches:
            raise ValueError(f"No target sketches found in {target_sketches_path}")

        #low_action = np.array([-5.0, -5.0, 0.0], dtype=np.float32)
        #high_action = np.array([5.0, 5.0, 1.0], dtype=np.float32)
        #self.action_space = spaces.Box(low=low_action, high=high_action, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Dict({
            "canvas": spaces.Box(low=0, high=255, shape=self.canvas_size, dtype=np.uint8),
            "target_sketch": spaces.Box(low=0, high=255, shape=self.canvas_size, dtype=np.uint8),
            "pen_position_mask": spaces.Box(low=0, high=255, shape=self.canvas_size, dtype=np.uint8),
        })

        self.canvas = None
        self.target_sketch = None
        self.cursor = [0, 0]
        self.is_pen_down = False

        self.window = None
        self.clock = None
        self.canvas_surface = None  # Pygame surface for the drawing canvas
        self.target_sketch_surface = None  # Pygame surface for the target sketch

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
                    sketches.append(np.array(sketch))
                except Exception as e:
                    print(f"Error loading sketch {filepath}: {e}")
        print(f"Loaded {len(sketches)} target sketches.")
        return sketches

    def _get_obs(self):
        pen_position_mask = np.zeros(self.canvas_size, dtype=np.uint8)
        pen_position_mask[self.cursor[1], self.cursor[0]] = 255
        return {
            "canvas": self.canvas.copy(),
            "target_sketch": self.target_sketch.copy(),
            "pen_position_mask": pen_position_mask,
        }

    def _get_info(self):
        return {"similarity": calculate_similarity(self.canvas, self.target_sketch)}

    def _decode_action(self, action):
        """
        Decodes a discrete action (0-17) into dx, dy, and pen_state.
        - Actions 0-8: Pen is UP.
        - Actions 9-17: Pen is DOWN.

        The 9 movements correspond to a 3x3 grid around the cursor.
        """
        # Determine pen state
        is_pen_down = action >= 9

        # Determine movement
        sub_action = action % 9
        dx = (sub_action % 3) - 1  # Results in -1, 0, or 1
        dy = (sub_action // 3) - 1  # Results in -1, 0, or 1

        return dx, dy, is_pen_down

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.hp = self.max_hp
        self.canvas = np.full(self.canvas_size, 255, dtype=np.uint8)

        self.target_sketch = random.choice(self.target_sketches)
        self.cursor = self._find_starting_point(self.target_sketch)
        self.is_pen_down = False
        self.last_pixel_similarity = calculate_pixel_similarity(self.canvas, self.target_sketch)

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

        #dx, dy, pen_action_logit = action
        #dx = int(np.round(dx))
        #dy = int(np.round(dy))

        #new_pen_state = bool(np.round(np.clip(pen_action_logit, 0, 1)))
        dx, dy, self.is_pen_down = self._decode_action(action)

        reward = 0.0
        terminated = False
        truncated = False

        prev_cursor_x, prev_cursor_y = self.cursor[0], self.cursor[1]

        prev_canvas_state = self.canvas.copy()

        self.cursor[0] = np.clip(self.cursor[0] + dx, 0, self.canvas_size[0] - 1)
        self.cursor[1] = np.clip(self.cursor[1] + dy, 0, self.canvas_size[1] - 1)

        if self.is_pen_down:
            #draw_line_on_canvas(self.canvas, prev_cursor_x, prev_cursor_y,
            #                    self.cursor[0], self.cursor[1], color=0, brush_size=1)
            #points = self._get_line_pixels(prev_cursor_x, prev_cursor_y, self.cursor[0], self.cursor[1])

            if self.is_pen_down:
                x, y = self.cursor
                if self.canvas[y, x] == 255:
                    self.canvas[y, x] = 0
                    if self.target_sketch[y, x] == 0:
                        reward += 0.1
                    else:
                        self.hp -= 1
                    #    reward -= 0.01
                else:
                    self.hp -= 1
                    reward -= 0.005

            #for px, py in points:
            #   if prev_canvas_state[py, px] == 255:
            #       if self.target_sketch[py, px] == 0:
            #           reward += 0.1
            #       else:
            #           reward -= 0.01

        # if new_pen_state != self.is_pen_down:
        #     self.is_pen_down = new_pen_state
        #     if self.is_pen_down and self.target_sketch[self.cursor[1], self.cursor[0]] == 0:
        #         reward += 0.02
        #     elif not self.is_pen_down and self.target_sketch[self.cursor[1], self.cursor[0]] == 255:
        #         reward += 0.02
        current_pixel_similarity = calculate_pixel_similarity(self.canvas, self.target_sketch)
        pixel_similarity_reward = current_pixel_similarity - self.last_pixel_similarity
        self.last_pixel_similarity = current_pixel_similarity  # 更新相似度记录

        # 乘以一个权重来调整这个奖励的影响力
        reward += pixel_similarity_reward * 5.0
        reward -= 0.001
        if self.current_step >= self.max_steps or self.hp <= 0:
            truncated = True

        # current_similarity = calculate_similarity(self.canvas, self.target_sketch)
        if current_pixel_similarity > 0.9:
            reward += 100.0
            terminated = True
            print(f"Task completed! Similarity: {current_pixel_similarity:.4f}")
        #reward += current_similarity
        # if truncated or terminated:
        #     shape_distance = calculate_shape_similarity_distance(self.canvas, self.target_sketch)
        #     final_shape_reward = 10.0 * np.exp(-5.0 * shape_distance)
        #     reward += final_shape_reward
        #
        #     print(f"Final Shape Reward: {final_shape_reward:.4f}")

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode: self.render()

        return observation, reward, terminated, truncated, info

    def _get_line_pixels(self, x1, y1, x2, y2):
        """Bresenham's Line Algorithm implementation to get pixels on a line."""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points

    def _find_starting_point(self, sketch_array):
        foreground_pixels = np.argwhere(sketch_array == 0)

        if foreground_pixels.size == 0:
            return [self.canvas_size[0] // 2, self.canvas_size[1] // 2]

        sorted_indices = np.lexsort((foreground_pixels[:, 1], foreground_pixels[:, 0]))
        top_left_pixel = foreground_pixels[sorted_indices[0]]

        return [top_left_pixel[1], top_left_pixel[0]]

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
            cursor_radius = 3
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