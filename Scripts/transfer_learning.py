from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.optimizers import RMSprop
from Scripts import constants as c
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import keras

# Clear session and instantiate model
keras.backend.clear_session()

# Load images & labels
cells = np.load(c.cells_path)
labels = np.load(c.labels_path)

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

# Instantiate VGG16 without top layers
vgg_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=c.input_shape)

# Use dictionary to map layer names to the VGG layers
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# Getting output tensor of the last VGG layer that we want to include
x = layer_dict['block2_pool'].output

# Add fully connected layers on top of VGG16
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# Create new model
model = Model(input=vgg_model.input, output=x)

# Freeze bottom layers of VGG model
for layer in model.layers[:7]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(train_x, train_y,
                    batch_size=600,
                    epochs=150,
                    validation_data=(val_x, val_y),
                    shuffle=False)

# Save model & weights
model.save_weights("1Model_weights.h5")
model.save("1Model.h5")

# Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
