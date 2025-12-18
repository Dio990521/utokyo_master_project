import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize


def get_active_endpoints(target, canvas):
    """
    计算当前'未完成'线条的端点坐标。
    返回: (y_coords, x_coords) 的 tuple，可以直接用于索引赋值
    """
    # 1. 计算差异 mask (1=未完成, 0=已完成/背景)
    # 阈值 0.05 过滤掉微小误差
    diff = (canvas - target) > 0.05

    if not np.any(diff):
        return None

    # 2. 骨架化：提取未完成部分的中心线
    # 32x32图像上这个操作非常快 (<1ms)
    skeleton = skeletonize(diff).astype(np.uint8)

    # 3. 寻找端点
    # 使用卷积计算邻居数
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    neighbors = cv2.filter2D(skeleton, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # 端点定义：本身是骨架(1) 且 周围邻居数 <= 1
    # (<=1 是为了包含孤立点的情况，防止死锁)
    endpoints_mask = (skeleton == 1) & (neighbors <= 1)

    # 如果没有端点（例如闭合圆环），退化为返回所有骨架点
    # 这样 Agent 会被引导去圆环上的任意一点切入
    if not np.any(endpoints_mask):
        return np.where(skeleton == 1)

    return np.where(endpoints_mask == 1)

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


def find_starting_point(sketch: np.ndarray):
    # Find any ink pixel (Black or Grey)
    foreground_pixels = np.argwhere(sketch < 0.99)
    if len(foreground_pixels) == 0: return [0, 0]
    random_index = random.randint(0, len(foreground_pixels) - 1)
    random_pixel_yx = foreground_pixels[random_index]
    #return [random_pixel_yx[1], random_pixel_yx[0]]
    first_pixel_yx = foreground_pixels[0]
    return [first_pixel_yx[1], first_pixel_yx[0]]