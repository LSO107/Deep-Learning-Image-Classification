from PIL import Image
from Scripts import constants as c
import numpy as np
import cv2
import os

# File paths for data
infected = os.listdir("../cell_images/Parasitized/")
uninfected = os.listdir("../cell_images/Uninfected/")

# Define arrays to store images & labels
images = []
labels = []

print('Starting image preprocessing...')

# label and resize data
for i in infected:
    try:
        image = cv2.imread("cell_images/Parasitized/" + i)
        image_array = Image.fromarray(image, 'RGB')
        resize_img = image_array.resize(c.image_size)
        images.append(np.array(resize_img))
        labels.append(1)
    except AttributeError:
        print('End of infected images')
for i in uninfected:
    try:
        image = cv2.imread("cell_images/Uninfected/" + i)
        image_array = Image.fromarray(image, 'RGB')
        resize_img = image_array.resize(c.image_size)
        images.append(np.array(resize_img))
        labels.append(0)
    except AttributeError:
        print('End of uninfected images')

# Convert to numpy arrays
cells = np.array(images)
labels = np.array(labels)

print('Converted data to numpy arrays')

# Save numpy arrays
np.save('Data/Cells', cells)
np.save('Data/Labels', labels)

print('Cells.npy and Labels.npy saved in Data folder')

print("Cells:", cells.shape, "Labels:", labels.shape)
