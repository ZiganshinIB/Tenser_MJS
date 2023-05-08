# Press the green button in the gutter to run the script.
import config
import models_tenser
from DataW import DataIMG



def step0():
    x_data, y_data = DataIMG.get_all_data_img(config.FOLDER, 'L', "R")
    DataIMG.save_data(x_data, y_data)

def step1():
    input_shape = list(X_train.shape)[1:]
    output_number = y_data.max()
    list_layers = ['cn64',
                   'cn64',

                   'mp',

                   'cn128',
                   'cn128',

                   'mp',

                   'cn256',

                   'mp',

                   'cn256',

                   'mp',

                   'cn512',

                   'f',
                   'd1000',
                   ':20',
                   "d100"]

    CNN_model = models_tenser.MyModelsCNN(input_shape=input_shape,
                                          list_layers=list_layers,
                                          output_number=output_number + 1)
    return CNN_model


def step2(CNN_model):
    CNN_model.fit(X_train, Y_train, batch_size=25, epochs=50)
    CNN_model.save("./models/test1")


def step3(CNN_model: models_tenser.MyModelsCNN):
    CNN_model.load_other("./models/mymodel_3")
    list_update_layers = \
        [
         ':20',
         "d100",
        ]
    CNN_model.update_model(froze_layers_number=13, list_update_layers_model=list_update_layers, learning_rate=.001,
                           train_layer_number=0)
    CNN_model.fit(x_train=X_train, y_train=Y_train,batch_size=15, epochs=12)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

if __name__ == '__main__':
    x_data, y_data = DataIMG.load_x_y()
    x_data, y_data = DataIMG.random_data(x_data, y_data)
    X_train, X_test, Y_train, Y_test = DataIMG.split_train_test(x_data, y_data, 0.2)

    CNN_model = step1()
    step3(CNN_model)
