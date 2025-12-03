import random
import numpy as np
from scipy.ndimage import distance_transform_edt


def calculate_f1_score(precision, recall):
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_dynamic_distance_map(target_sketch: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    # 0.0 is target, 1.0 is background
    # We want distance to "Remaining Black/Grey Pixels"
    # [UPDATED] logic: Any pixel where canvas > target needs work.
    remaining_work_map = np.where(
        canvas > target_sketch,  # Features not yet fully dark enough
        0.0,  # Feature (Needs work)
        1.0  # Background or Completed
    )
    return distance_transform_edt(remaining_work_map)


def calculate_metrics(target, canvas, canvas_size):
    # Tolerance for floating point comparison
    tol = 0.01

    # Black: 0.0, Grey: (0.0, 1.0), White: 1.0
    is_target_black = target <= tol
    is_target_white = target >= (1.0 - tol)
    is_target_grey = (~is_target_black) & (~is_target_white)

    pixel_diff = np.abs(target - canvas)
    is_match = pixel_diff <= tol

    # TP: Target is Ink (Black or Grey) AND Canvas Matches
    # Correctly painted Black pixels
    tp_black = np.sum(is_target_black & is_match)
    total_target_black = np.sum(is_target_black)

    # Correctly painted Grey pixels
    tp_grey = np.sum(is_target_grey & is_match)
    total_target_grey = np.sum(is_target_grey)

    # Correctly left White pixels (TN)
    tn = np.sum(is_target_white & is_match)
    total_target_white = np.sum(is_target_white)

    # FP: Canvas has Ink (or darker value) where Target is White
    # Or Canvas is darker than Target (Overshoot)
    fp = np.sum(is_target_white & (canvas < (1.0 - tol)))

    recall_black = (tp_black / total_target_black) if total_target_black > 0 else 1.0
    recall_grey = (tp_grey / total_target_grey) if total_target_grey > 0 else 1.0
    total_ink_target = total_target_black + total_target_grey
    recall_all = ((tp_black + tp_grey) / total_ink_target) if total_ink_target > 0 else 1.0
    recall_white = (tn / total_target_white) if total_target_white > 0 else 0.0

    # Precision: (Correct Ink) / (Total Canvas Ink)
    # Total Canvas Ink = Pixels < 1.0
    is_canvas_ink = canvas < (1.0 - tol)
    total_canvas_ink = np.sum(is_canvas_ink)

    # "Correct Ink" here means match on Black or Grey
    tp_total = tp_black + tp_grey
    precision = (tp_total / total_canvas_ink) if total_canvas_ink > 0 else 0.0

    # Simple Pixel Similarity (1 - avg_diff)
    current_pixel_similarity = 1.0 - np.mean(pixel_diff)

    return recall_black, recall_grey, recall_all, recall_white, precision, current_pixel_similarity


def calculate_reward_map(target_sketch, reward_on_target=0.1, reward_near_target=0.0, reward_far_target=-0.1,
                         near_distance=2):
    if reward_near_target == reward_far_target:
        reward_map = np.where(
            np.isclose(target_sketch, 1.0),  # Background
            reward_far_target,
            reward_on_target  # Ink (Grey or Black)
        )
        return reward_map.astype(np.float32)

    distance_map = distance_transform_edt(target_sketch)  # Uses value as distance? No, edt works on boolean.
    # For edt on grayscale, we usually threshold.
    binary_sketch = np.where(target_sketch < 0.99, 0, 1)  # Ink is 0
    distance_map = distance_transform_edt(binary_sketch)

    reward_map = np.full(target_sketch.shape, reward_far_target, dtype=np.float32)
    reward_map[(distance_map <= near_distance) & (distance_map > 0)] = reward_near_target
    reward_map[distance_map == 0] = reward_on_target
    return reward_map


def find_starting_point(sketch: np.ndarray):
    # Find any ink pixel (Black or Grey)
    foreground_pixels = np.argwhere(sketch < 0.99)
    if len(foreground_pixels) == 0: return [0, 0]
    random_index = random.randint(0, len(foreground_pixels) - 1)
    random_pixel_yx = foreground_pixels[random_index]
    return [random_pixel_yx[1], random_pixel_yx[0]]