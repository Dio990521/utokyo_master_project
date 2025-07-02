import cv2
import numpy as np

# 加载并二值化
img1 = cv2.imread("smile.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("sketches/line_20x20.png", cv2.IMREAD_GRAYSCALE)

_, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)
_, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 取最大的轮廓（假设只有一个物体）
cnt1 = max(contours1, key=cv2.contourArea)
cnt2 = max(contours2, key=cv2.contourArea)

# 计算轮廓相似度
similarity = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)
print(f"Contour Similarity (lower is better): {similarity:.4f}")