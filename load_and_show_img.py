import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import time

data = [
    '1670848078-cornell_box_linear_data.npz',
    '1670848645-cornell_box_linear_data.npz',
    '1670848685-cornell_box_linear_data.npz',
    '1670866387-cornell_box_linear_data.npz',
    '1670866419-cornell_box_linear_data.npz',
    '1670866484-cornell_box_linear_data.npz',
    '1670852590-cornell_box_linear_data.npz',
    '1670852602-cornell_box_linear_data.npz',
    '1670852616-cornell_box_linear_data.npz',
    '1670852682-cornell_box_linear_data.npz'
]

loaded = np.load(data[0])
img = np.zeros(loaded["image"].shape)


for d in data:
    loaded = np.load(d)
    img += loaded["image"]

i = ndimage.rotate(img, -90)
m = np.max(img)/5
i = np.power((i/m), 0.45)
i = np.clip(i, 0, 1)
fig = plt.imshow(i)
plt.pause(60)