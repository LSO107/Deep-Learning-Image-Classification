from keras.models import load_model
import numpy as np
import cv2

# load model
model = load_model("classifier.h5")

# image path
img_path = 'dataset/single_prediction/cat_or_dog_1.png'

# load a single image
img = cv2.imread('dataset/single_prediction/cat_or_dog_2.png')
img = cv2.resize(img, (64, 64))
img = np.reshape(img, [1, 64, 64, 3])

# check prediction
pred = model.predict(img)

if pred[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# print result
print(prediction)
