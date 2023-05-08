import keras
from keras import Input, callbacks
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

class MyModelsCNN:

    def __init__(self, input_shape, list_layers:list[str], output_number):
        self.model = keras.Sequential()
        self.input_shape = input_shape
        self.output_number = output_number

        self.model.add(Input(shape=self.input_shape, name="input"))
        self.add_layers(list_layers)
        self.model.add(Dense(self.output_number, activation='softmax'))

        self.init_callbacks()
        self.compile()
        print(self.model.summary())

    def add_layers(self, list_layers:list[str], index_update:int=1):
        number_c = 0
        number_d = 0
        for m in list_layers:
            if m.startswith("cn"):
                number_c += 1
                kernel = int(m[2:])
                print(f"Добавляю слой conv c {kernel} шаблонами")
                self.model.add(Conv2D(kernel, (3, 3), padding="same", activation="relu", name=f'conv{index_update}-{number_c}'))
                continue
            if m.startswith("mp"):
                print("Добавляю слой MaxPooling2D")
                self.model.add(MaxPooling2D(pool_size=(2, 2)))
                continue
            if m.startswith("f"):
                print("Добавляю слой Flatten")
                self.model.add(Flatten())
                continue
            if m.startswith("d"):
                number_d += 1
                dense = int(m[1:])
                print(f"Добавляю слой Dense c {dense} нейронами")
                self.model.add(Dense(dense, activation='relu', name=f"hide{index_update}-{number_d}"))
                continue
            if m.startswith(":"):
                dr = float(int(m[1:]) / 100)
                print(f"Добавляю слой Dropout c {dr} удалением")
                self.model.add(Dropout(dr))
                continue

    def init_callbacks(self):
        self.my_callbacks = [
            callbacks.EarlyStopping(monitor="loss", min_delta=0.01, patience=2, verbose=1),
            callbacks.ModelCheckpoint(filepath='mymodel_{epoch}', save_best_only=True, monitor="loss", verbose=1),
        ]

    def add_callbacks(self, other_callbacks: callbacks.Callback):
        self.my_callbacks.append(other_callbacks)

    def compile(self, learning_rate=.01):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])

    def fit(self, x_train, y_train, batch_size=15, epochs=12, validation_split=0.2):
        self.history = self.model.fit(x=x_train, y=y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_split=validation_split,
                                      callbacks=self.my_callbacks)

    def save(self, file_name):
        self.model.save(file_name)

    def load_other(self, file_name):
        other_model = keras.models.load_model(file_name)
        print(other_model.summary())
        self.model = other_model

    def update_model(self, froze_layers_number:int, list_update_layers_model:list[str], learning_rate=0.001, train_layer_number=0):
        other_model = self.model
        self.model = keras.Sequential()
        self.model.add(Input(shape=self.input_shape, name="input"))
        for i in range(froze_layers_number):
            self.model.add(other_model.get_layer(index=i))
            print(f"Не буду обучать {other_model.get_layer(index=i).name}")
            other_model.get_layer(index=i).trainable = False
        if train_layer_number > 0:
            for i in range(froze_layers_number, froze_layers_number+train_layer_number):
                self.model.add(other_model.get_layer(index=i))
                print(f"Буду обучать {other_model.get_layer(index=i).name}")
        self.add_layers(list_update_layers_model, index_update=3)
        self.model.add(Dense(self.output_number, activation='softmax'))
        self.init_callbacks()
        self.compile(learning_rate=learning_rate)
        print(self.model.summary())
