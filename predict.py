"""
Predict image: python predict.py -i/--image {IMAGE_PATH}
"""

import warnings
import tensorflow.keras as keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import argparse
import pathlib
warnings.filterwarnings("ignore")


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


categories = ['Babi', 'Calimero', 'Chrysanthemum', 'Hydrangeas', 'Lisianthus', 'Pingpong', 'Rosy', 'Tana']
custom_objects = {"f1_m": f1_m}
model = keras.models.load_model('saved/model_AlexNet.h5', custom_objects=custom_objects)


def predict_image(path):
    img = keras.utils.load_img(path, target_size=(160, 160))
    img.show()
    img = keras.utils.img_to_array(img)
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction_label = categories[np.argmax(prediction)]
    print(prediction_label)


parser = argparse.ArgumentParser(description='Provide Image')
parser.add_argument("-i", "--image", type=str, help="Image path here")
args = parser.parse_args()

if args.image is None:
    print('Please provide image.')
else:
    predict_image(args.image)