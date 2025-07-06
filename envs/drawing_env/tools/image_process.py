import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

def visualize_obs(obs):
    canvas = obs[0]
    target = obs[1]
    pen_mask = obs[2]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(canvas, cmap='gray')
    axs[0].set_title("Canvas")

    axs[1].imshow(target, cmap='gray')
    axs[1].set_title("Target Sketch")

    axs[2].imshow(pen_mask, cmap='gray')
    axs[2].set_title("Pen Position")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def find_starting_point(sketch):
    foreground_pixels = np.argwhere(sketch == 0)

    if foreground_pixels.size == 0:
        return [self.canvas_size[0] // 2, self.canvas_size[1] // 2]

    sorted_indices = np.lexsort((foreground_pixels[:, 1], foreground_pixels[:, 0]))
    top_left_pixel = foreground_pixels[sorted_indices[0]]

    return [top_left_pixel[1], top_left_pixel[0]]

def calculate_pixel_similarity(canvas, target_sketch):
    total_target_pixels = np.sum(target_sketch == 0)

    if total_target_pixels == 0:
        return 1.0

    overlapping_pixels = np.sum((canvas == 0) & (target_sketch == 0))
    similarity = overlapping_pixels / total_target_pixels

    return similarity

def calculate_shape_similarity_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    img1_inv = cv2.bitwise_not(image1)
    img2_inv = cv2.bitwise_not(image2)

    contours1, _ = cv2.findContours(img1_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours1 or not contours2:
        return 10.0

    main_contour1 = max(contours1, key=cv2.contourArea)
    main_contour2 = max(contours2, key=cv2.contourArea)

    distance = cv2.matchShapes(main_contour1, main_contour2, cv2.CONTOURS_MATCH_I1, 0.0)

    return distance

def calculate_block_reward(canvas, target_sketch, block_size):
    canvas_h, canvas_w = canvas.shape
    score = 0
    num_blocks = 0

    for y in range(0, canvas_h, block_size):
        for x in range(0, canvas_w, block_size):
            num_blocks += 1

            canvas_block = canvas[y:y + block_size, x:x + block_size]
            target_block = target_sketch[y:y + block_size, x:x + block_size]

            canvas_has_black = np.any(canvas_block == 0)
            target_has_black = np.any(target_block == 0)

            if (canvas_has_black and target_has_black) or \
                    (not canvas_has_black and not target_has_black):
                score += 1

    return score / num_blocks if num_blocks > 0 else 0

def draw_line_on_canvas(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: int, brush_size: int = 1):

    img = Image.fromarray(canvas, 'L') # 'L' for grayscale
    draw = ImageDraw.Draw(img)

    draw.line((x1, y1, x2, y2), fill=color, width=brush_size)

    canvas[:] = np.array(img)