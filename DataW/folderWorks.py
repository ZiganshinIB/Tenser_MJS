import os
import numpy as np
import cv2


def get_direct_images_to_array(path, file_end, inclusion, exception, id_name=None,):
    images = []
    marks = []

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(file_end) and \
               (inclusion in file) and not (exception in file):
                filepath = os.path.join(subdir, file)
                img = cv2.imread(filepath)
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img)
                img = img/255
                img = np.reshape(img, (224, 224, 1))
                images.append(img)
                marks.append(id_name)
                print(f"\tAdd {filepath} image shape: {img.shape}")
    return (images, marks)


def __get_condition_inclusion__(inclusion):
    return lambda file, file_end: file.endswith(file_end) and (inclusion in file)


def __get_condition_exception__(exception):
    return lambda file, file_end: file.endswith(file_end) and not (exception in file)


def __get_condition_inclusion_and_exception__(inclusion, exception):
    return lambda file, file_end: file.endswith(file_end) and \
                                  (inclusion in file) and not \
                                  (exception in file)


