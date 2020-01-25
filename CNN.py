from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
import keras
from matplotlib import pyplot

keras.backend.clear_session()

# create model
model = Sequential()

# add layers to neural network
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# preprocess the dataset
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(128, 128),
                                                     batch_size=32,
                                                     class_mode='binary')

test_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')

# fit the model
model.fit_generator(training_dataset,
                    steps_per_epoch=2000,
                    epochs=50,
                    validation_data=test_dataset,
                    validation_steps=800)

# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# save the model
model.save("Model.h5")
