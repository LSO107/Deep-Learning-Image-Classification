import os
import numpy as np
from PIL import Image

folder = "/cell_images"
images = sorted(os.listdir(folder))

image_array = []

for image in image_array:
    im = Image.open(folder + image)
    image_array.append(np.asarray(im))

    image_array = np.array(image_array)
    print(image_array.shape)
