import random
import numpy as np
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

def calculate_f1_score(precision, recall):
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_metrics(tp, fp, tn, fn, canvas_size):
    total_target_black = tp + fn
    current_recall_black = (tp / total_target_black) if total_target_black > 0 else 0.0

    total_target_white = tn + fp
    current_recall_white = (tn / total_target_white) if total_target_white > 0 else 0.0

    total_canvas_black = tp + fp
    current_precision_black = (tp / total_canvas_black) if total_canvas_black > 0 else 0.0

    current_pixel_similarity = (tp + tn) / canvas_size
    return current_recall_black, current_recall_white, current_precision_black, current_pixel_similarity


def calculate_metrics_grey(target, canvas):
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

def find_starting_point(sketch: np.ndarray):
    # Find any ink pixel (Black or Grey)
    foreground_pixels = np.argwhere(sketch < 0.99)
    if len(foreground_pixels) == 0: return [0, 0]
    random_index = random.randint(0, len(foreground_pixels) - 1)
    random_pixel_yx = foreground_pixels[random_index]
    return [random_pixel_yx[1], random_pixel_yx[0]]