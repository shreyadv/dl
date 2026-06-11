import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

noise = np.random.normal(0, 25, img.shape)
noisy = img.astype(np.float32) + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
median = cv2.medianBlur(noisy, 5)

images = [
    img_rgb,
    cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
]

titles = ["Original", "Noisy", "Gaussian Blur", "Median Filter"]

plt.figure(figsize=(20, 15))

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
