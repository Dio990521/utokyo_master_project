import random
import numpy as np
from scipy.ndimage import distance_transform_edt

def calculate_f1_score(precision, recall):
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_dynamic_distance_map(target_sketch: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    # 0.0 is target, 1.0 is background
    # We want distance to "Remaining Black Pixels"
    remaining_work_map = np.where(
        (np.isclose(target_sketch, 0.0)) & (np.isclose(canvas, 1.0)),
        0.0, # Features (Target)
        1.0  # Background
    )
    return distance_transform_edt(remaining_work_map)

def calculate_metrics(tp, fp, tn, fn, canvas_size):
    total_target_black = tp + fn
    current_recall_black = (tp / total_target_black) if total_target_black > 0 else 0.0

    total_target_white = tn + fp
    current_recall_white = (tn / total_target_white) if total_target_white > 0 else 0.0

    total_canvas_black = tp + fp
    current_precision_black = (tp / total_canvas_black) if total_canvas_black > 0 else 0.0

    current_pixel_similarity = (tp + tn) / canvas_size
    return current_recall_black, current_recall_white, current_precision_black, current_pixel_similarity

def calculate_reward_map(target_sketch, reward_on_target=0.1, reward_near_target=0.0, reward_far_target=-0.1, near_distance=2):
    if reward_near_target == reward_far_target:
        reward_map = np.where(
            np.isclose(target_sketch, 0),
            reward_on_target,
            reward_far_target
        )
        return reward_map.astype(np.float32)
    distance_map = distance_transform_edt(target_sketch)
    reward_map = np.full(target_sketch.shape, reward_far_target, dtype=np.float32)
    reward_map[(distance_map <= near_distance) & (distance_map > 0)] = reward_near_target
    reward_map[distance_map == 0] = reward_on_target
    return reward_map

def find_starting_point(sketch: np.ndarray):
    foreground_pixels = np.argwhere(np.isclose(sketch, 0.0))
    if len(foreground_pixels) == 0: return [0, 0]
    random_index = random.randint(0, len(foreground_pixels) - 1)
    random_pixel_yx = foreground_pixels[random_index]
    return [random_pixel_yx[1], random_pixel_yx[0]]