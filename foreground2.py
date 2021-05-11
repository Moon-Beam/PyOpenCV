import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img/unnamed.jpg')
imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = np.zeros(imgRBG.shape[:2],  np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (20, 50, 562, 364)
#rect = (20, 20, 250, 144)
#rect = (193, 67, 650, 630)
#rect = (50, 50, 1000, 1000)


cv2.grabCut(imgRBG, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
imgRBG = imgRBG*mask2[:, :, np.newaxis]

plt.imshow(imgRBG)
plt.colorbar()
plt.show()
