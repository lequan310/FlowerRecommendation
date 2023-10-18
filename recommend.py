"""
Recommend images: python recommend.py -i/--image {IMAGE_PATH}
"""

import os, hashlib, random
import pandas as pd
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import argparse
import tensorflow.keras as keras
from keras import backend as K
from keras import Model
from sklearn.manifold import TSNE

PROJECT_DIR = ''
DATASET_PATH = PROJECT_DIR + 'dataset/Flowers/'

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

kmeans = joblib.load("saved/model_kmeans.joblib")
feat = joblib.load("saved/feat.data")
groups = joblib.load("saved/groups.data")


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


custom_objects = {"f1_m": f1_m}
model_alexnet = keras.models.load_model('saved/model_AlexNet.h5', custom_objects=custom_objects)
feature_extractor = Model(inputs=model_alexnet.inputs, outputs=model_alexnet.layers[-2].output)
tsne = TSNE(n_components=2, random_state=0)


def recommend_similar_images(image_path):
    image = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image.show()
    image = keras.utils.img_to_array(image)
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    features = feature_extractor.predict(image)
    temp_feat = feat
    temp_feat = np.append(temp_feat, np.array([features]), axis=0)
    temp_feat = temp_feat.reshape(-1, 1000)
    temp_feat = tsne.fit_transform(temp_feat)
    prediction = kmeans.predict(temp_feat)
    fig, axes = plt.subplots(1, 10, figsize=(15, 5))

    for i in range(10):
        random_image = random.choice(groups[prediction[-1]])
        img = cv2.imread(DATASET_PATH + random_image)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(random_image)

        axes[i].imshow(image_rgb)
        axes[i].axis('off')
    plt.show()


parser = argparse.ArgumentParser(description='Provide Image')
parser.add_argument("-i", "--image", type=str, help="Image path here")
args = parser.parse_args()

if args.image is None:
    print('Please provide image.')
else:
    recommend_similar_images(args.image)