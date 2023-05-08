import numpy as np
import os
from .folderWorks import get_direct_images_to_array


def get_all_data_img(path, inclusion, exception):
    """
    Получить все изображении в виде массива
    :param path: путь где храниятся директории с классами
    :param inclusion: файл содержить
    :param exception: файл не соддержить
    :return: изображение и индекс директории
    """
    x = []
    y = []
    for i, direct_name in enumerate(os.listdir(path)):
        res = get_direct_images_to_array(os.path.join(path, direct_name),
                                         ".png",
                                         inclusion,
                                         exception,
                                         i)
        x += res[0]
        y += res[1]
    x, y = np.array(x), np.array(y)
    print(f"размер X {x.shape}")
    print(f"размер Y {y.shape}")
    return np.array(x), np.array(y)


def random_data(x_data, y_data):
    """
    Перемешивает элементы
    :param x_data: X
    :param y_data: Y
    :return: Перемешинные элементы
    """
    indices = np.random.permutation(len(x_data))
    return x_data[indices], y_data[indices]


def split_train_test(x_data, y_data, test_size):
    num_test = int(len(x_data) * test_size)
    indices = np.random.permutation(len(x_data))
    # Получаем индексы для обучающей и тестовой выборок
    train_indices = indices[:-num_test]
    test_indices = indices[-num_test:]
    # Разделяем массивы X и Y на обучающую и тестовую выборки
    X_train, X_test = x_data[train_indices], x_data[test_indices]
    Y_train, Y_test = y_data[train_indices], y_data[test_indices]
    return X_train, X_test, Y_train, Y_test

def save_data(x_data:np.ndarray, y_data:np.ndarray):
    np.save("Data_X_Y/X.npy", x_data)
    np.save("Data_X_Y/Y.npy", y_data)


def load_x_y():
    x_data = np.load("Data_X_Y/X.npy")
    y_data = np.load("Data_X_Y/Y.npy")
    return x_data, y_data

