from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from Scripts import constants as c
import matplotlib.pyplot as plt
import numpy as np
import keras

# Clear session and instantiate model
keras.backend.clear_session()
model = Sequential()

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

# Neural network
model.add(Conv2D(32, (3, 3), input_shape=c.input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
testing_datagen = ImageDataGenerator(rescale=1. / 255)

training_dataset = train_datagen.flow(train_x, train_y, batch_size=32)
validation_dataset = validation_datagen.flow(val_x, val_y, batch_size=32)
testing_dataset = validation_datagen.flow(val_x, val_y, batch_size=32)

# Add callbacks to prevent overfitting
es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=2,
                   verbose=0,
                   mode='auto')

checkpoint = ModelCheckpoint("Model.h5")

# Perform backpropagation and update weights in model
history = model.fit_generator(training_dataset,
                              epochs=50,
                              validation_data=validation_dataset,
                              callbacks=[es, checkpoint])

# Save model & weights
model.save_weights("Model_weights.h5")
model.save("Model.h5")

# Plot accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
