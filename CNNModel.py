import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers

class CNNModel:

    def __init__(self, name="j"):
        self.name=name

    def fit(self, data):
        X_train, y_train = data
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        self.model=model
        self.model.fit(np.array(X_train), np.array(y_train), epochs=10)
        return self.model

    def predict(self, ):
        return self

    def fit_predict(self):
        return 0


