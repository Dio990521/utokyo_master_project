import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
import cv2


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

def draw_line_on_canvas(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: int, brush_size: int = 1):

    img = Image.fromarray(canvas, 'L') # 'L' for grayscale
    draw = ImageDraw.Draw(img)

    draw.line((x1, y1, x2, y2), fill=color, width=brush_size)

    canvas[:] = np.array(img)