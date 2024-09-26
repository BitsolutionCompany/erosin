import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('4.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((100,100), np.uint8)
erosin = cv.erode(img, kernel, iterations = 1)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.ravel()

ax[0].set_title('Original Picture')
ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_axis_off()

ax[1].set_title('Transformed Picture')
ax[1].imshow(erosin, cmap=plt.cm.gray)
ax[1].set_axis_off()

plt.tight_layout()
plt.show()