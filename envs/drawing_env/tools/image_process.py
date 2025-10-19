import random

import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def calculate_penalty_map(target_sketch, safe_distance=2):

    distance_map = distance_transform_edt(target_sketch)

    # Penalty map is True where distance is greater than the safe_distance.
    penalty_map = distance_map > safe_distance
    return penalty_map

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

def calculate_iou_similarity(target_sketch, canvas, black_pixel_value=0) -> float:
    target_bool = (target_sketch == black_pixel_value)
    canvas_bool = (canvas == black_pixel_value)

    intersection = np.sum(target_bool & canvas_bool)

    union = np.sum(target_bool | canvas_bool)

    if union == 0:
        return 1.0

    iou = intersection / union
    return iou


def find_starting_point(sketch: np.ndarray):
    foreground_pixels = np.argwhere(np.isclose(sketch, 0.0))

    if len(foreground_pixels) == 0:
        return [0, 0]

    random_index = random.randint(0, len(foreground_pixels) - 1)
    random_pixel_yx = foreground_pixels[random_index]

    return [random_pixel_yx[1], random_pixel_yx[0]]


def calculate_qualified_block_similarity(canvas, target_sketch, block_size):
    canvas_h, canvas_w = canvas.shape
    score = 0
    total_blocks = 0

    for y in range(0, canvas_h, block_size):
        for x in range(0, canvas_w, block_size):
            total_blocks += 1
            canvas_block = canvas[y:y + block_size, x:x + block_size]
            target_block = target_sketch[y:y + block_size, x:x + block_size]
            block_iou = calculate_iou_similarity(target_block, canvas_block)
            score += block_iou

    return score / total_blocks if total_blocks > 0 else 0.0

def calculate_pixel_similarity(canvas, target_sketch):
    total_target_pixels = np.sum(target_sketch == 0)

    if total_target_pixels == 0:
        return 1.0

    overlapping_pixels = np.sum((canvas == 0) & (target_sketch == 0))
    similarity = overlapping_pixels / total_target_pixels

    return similarity

def load_image_as_array(image_path):
    # Convert to grayscale and binary (0: black, 255: white)
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    # Optional: thresholding to ensure binary format
    binary_image = np.where(image_array < 128, 0, 255).astype(np.uint8)

    return binary_image

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


def calculate_density_cap_reward(canvas_after, target_sketch, cursor_pos, block_size):
    x, y = cursor_pos
    h, w = target_sketch.shape

    half_size = block_size // 2
    x_start, x_end = max(0, x - half_size), min(w, x + half_size + 1)
    y_start, y_end = max(0, y - half_size), min(h, y + half_size + 1)

    target_block = target_sketch[y_start:y_end, x_start:x_end]
    canvas_block_after = canvas_after[y_start:y_end, x_start:x_end]

    num_black_target = np.sum(np.isclose(target_block, 0.0))

    num_black_canvas = np.sum(np.isclose(canvas_block_after, 0.0))

    if num_black_target >= num_black_canvas:
        return 0.1
    else:
        return -0.1