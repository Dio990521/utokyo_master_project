import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
import cv2


def calculate_shape_similarity_distance(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    使用 cv2.matchShapes 计算两个图像中主要轮廓的形状距离。
    返回值越小，形状越相似。
    """
    # cv2.findContours 需要二值化图像，并且是白底黑字。我们的输入是黑底白字，需要反转。
    # 我们假设输入图像是 0=黑色笔迹, 255=白色背景
    img1_inv = cv2.bitwise_not(image1)
    img2_inv = cv2.bitwise_not(image2)

    # 找到轮廓
    contours1, _ = cv2.findContours(img1_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(img2_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果任一图像没有轮廓（例如，空白画布），则认为它们非常不相似
    if not contours1 or not contours2:
        return 10.0  # 返回一个较大的距离值

    # 通常我们只比较最大的那个轮廓
    main_contour1 = max(contours1, key=cv2.contourArea)
    main_contour2 = max(contours2, key=cv2.contourArea)

    # 使用方法 'cv2.CONTOURS_MATCH_I1' 来计算形状距离
    distance = cv2.matchShapes(main_contour1, main_contour2, cv2.CONTOURS_MATCH_I1, 0.0)

    return distance

def calculate_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Jaccard (IoU) 。
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    binary_img1 = (image1 == 0).astype(int)
    binary_img2 = (image2 == 0).astype(int)

    intersection = np.sum(binary_img1 * binary_img2)
    union = np.sum((binary_img1 + binary_img2) > 0)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def draw_line_on_canvas(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: int, brush_size: int = 1):

    img = Image.fromarray(canvas, 'L') # 'L' for grayscale
    draw = ImageDraw.Draw(img)

    # 绘制线段，PIL的line函数可以直接设置宽度
    draw.line((x1, y1, x2, y2), fill=color, width=brush_size)

    # 将PIL图像转换回NumPy数组并更新画布
    canvas[:] = np.array(img)