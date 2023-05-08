import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
import keras
import tensorflow as tf
import config




def img_to_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #adaptiveThreshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 1)
    #getStructuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))
    # morphologyEx = cv2.morphologyEx(adaptiveThreshold, cv2.MORPH_CLOSE, getStructuringElement)
    return gray



def load_img(dir, inclusion, exception):
    images = []
    marks = []
    for folder_name in os.listdir(dir):
        mark = folder_name.partition('-')[0]
        print(mark)
        for subdir, dirs, files in os.walk(os.path.join(dir, folder_name)):
            for file in files:
                if file.endswith('.png') and (inclusion in file) and not (exception in file):
                    filepath = os.path.join(subdir, file)
                    img = cv2.imread(filepath)
                    img = cv2.resize(img, (224, 224))
                    img = img_to_binary(img)
                    img = np.array(img)
                    images.append(img)
                    marks.append(mark)
    images = np.array(images)
    marks = np.array(marks)
    return images, marks

exc1 = '1_5'
inc1 = 'L'
x_train, y_train = load_img(config.FOLDER, inc1, exc1)
print()

exc2 = 'nothing'
inc2 = 'L_1_5'
x_test, y_test = load_img(config.FOLDER, inc2, exc2)
