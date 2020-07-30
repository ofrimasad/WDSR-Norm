
dir_a = 'results/images_deconv/'
dir_b = 'results/images_norm_deconv/'

import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


directory = os.fsencode(dir_a)


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        a = Image.open(dir_a + filename, 'r')
        b = Image.open(dir_b + filename, 'r')

        a = np.array(a).astype(np.float32)
        b = np.array(b).astype(np.float32)

        diff = np.abs(a - b)

        plt.imshow(diff, interpolation='nearest')
        plt.show()
        continue
