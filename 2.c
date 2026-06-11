import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols = img.shape[:2]

M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

M_translate = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(img, M_translate, (cols, rows))

combined = cv2.warpAffine(rotated, M_translate, (cols, rows))

images = [
    img_rgb,
    cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(translated, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
]

titles = ["Original", "Rotated", "Translated", "Rotated + Translated"]

plt.figure(figsize=(25, 18))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i])
    plt.axis('off')

plt.show()
