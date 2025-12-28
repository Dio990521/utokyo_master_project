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

    is_target_ink = target < (1.0 - tol)
    is_target_white = target >= (1.0 - tol)
    is_canvas_ink = canvas < (1.0 - tol)

    tp_ink = np.sum(is_target_ink & is_canvas_ink)
    is_canvas_match_white = np.abs(target - canvas) <= tol
    tn = np.sum(is_target_white & is_canvas_match_white)

    total_target_ink = np.sum(is_target_ink)
    total_target_white = np.sum(is_target_white)

    recall_grey = (tp_ink / total_target_ink) if total_target_ink > 0 else 1.0
    recall_white = (tn / total_target_white) if total_target_white > 0 else 0.0

    total_canvas_ink = np.sum(is_canvas_ink)
    precision = (tp_ink / total_canvas_ink) if total_canvas_ink > 0 else 0.0

    pixel_diff = np.abs(target - canvas)
    current_pixel_similarity = 1.0 - np.mean(pixel_diff)

    return recall_grey, recall_white, precision, current_pixel_similarity


# def find_starting_point(sketch: np.ndarray):
#     # Find any ink pixel (Black or Grey)
#     foreground_pixels = np.argwhere(sketch < 0.99)
#     if len(foreground_pixels) == 0: return [0, 0]
# #     #print([foreground_pixels[0][1], foreground_pixels[0][0]])
#     return [3, 4]
# #     return [foreground_pixels[0][1], foreground_pixels[0][0]]
#     #random_index = random.randint(0, len(foreground_pixels) - 1)
#     #random_pixel_yx = foreground_pixels[random_index]
#     #return [random_pixel_yx[1], random_pixel_yx[0]]

def find_starting_point(sketch: np.ndarray):
    h, w = sketch.shape
    black = sketch < 0.99

    endpoints = []

    for y in range(h):
        for x in range(w):
            if not black[y, x]:
                continue

            # count black neighbors in 8-neighborhood
            neighbor_count = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if black[ny, nx]:
                            neighbor_count += 1

            if neighbor_count == 1:
                endpoints.append((x, y))

    if len(endpoints) > 0:
        return list(random.choice(endpoints))

    foreground_pixels = np.argwhere(black)
    if len(foreground_pixels) == 0:
        return [0, 0]

    y, x = foreground_pixels[0]
    return [x, y]