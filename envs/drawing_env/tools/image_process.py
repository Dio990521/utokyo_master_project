import numpy as np
from PIL import Image, ImageDraw

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