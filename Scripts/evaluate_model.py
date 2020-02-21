from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from keras.models import load_model
from keras.utils import plot_model
from Scripts import constants as c
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up path for GraphViz to work
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Load model
model = load_model("./Models/vgg16_1.h5")
# model = load_model("./Models/custom/Model.h5")

# Load images & labels
cells = np.load(os.path.join('./', c.cells_path))
labels = np.load(os.path.join('./', c.labels_path))

# Shuffle the entire dataset
n = np.arange(cells.shape[0])
np.random.shuffle(n)

# Update numpy files with shuffled data
cells = cells[n]
labels = labels[n]

# Split the dataset into train/validation/test
train_x, test_x, train_y, test_y = train_test_split(cells, labels, test_size=1 - c.train_ratio, shuffle=False)
val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=c.test_ratio / (c.test_ratio + c.val_ratio),
                                                shuffle=False)

# The amount of images in each set
print('Training data shape: ', train_x.shape)
print('Validation data shape: ', val_x.shape)
print('Testing data shape: ', test_x.shape)

# Generate model visualisation
print(model.summary())
plot_model(model, to_file='model.png')

# Plot graph to visualise predictions on testing data
plt.figure(figsize=(10, 10))
for i in range(49):
    if i == 4:
        plt.title('Label | Prediction')
    plt.subplot(7, 7, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(hspace=0.25, wspace=1)
    plt.imshow(test_x[i], cmap=plt.cm.binary)
    image = np.reshape(test_x[i], c.reshape_size)
    prediction = model.predict(image)

    if prediction[0][0] == 1:
        # + Positive / Infected
        pred = '+'
    else:
        # - Negative / Uninfected
        pred = '-'

    plt.xlabel('{} | {}'.format(c.class_names[test_y[i]], pred))
plt.show()
