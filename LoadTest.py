from keras.models import load_model
import numpy as np
import cv2

# load model
model = load_model("ImageClassificationModel.h5")

# image path
img_path = 'dataset/single_prediction'

# load a single image
img = cv2.imread(img_path + '/cat_or_dog_2.png')
img = cv2.resize(img, (64, 64))
img = np.reshape(img, [1, 64, 64, 3])

# check prediction
pred = model.predict(img)

if pred[0][0] == 1:
    prediction = 'Uninfected'
else:
    prediction = 'Parasitized'

# print result
print(prediction)
