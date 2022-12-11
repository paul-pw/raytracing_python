import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import time

loaded = np.load('1670752403-cornell_box_linear_data.npz')

print(loaded["image"])
img = loaded["image"]
i = ndimage.rotate(img, -90)
m = np.max(img)/50
i = np.power((i/m), 0.45)
i = np.clip(i, 0, 1)
fig = plt.imshow(i)
plt.pause(60)
