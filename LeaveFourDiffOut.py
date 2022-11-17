import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from ColorsData import *


class LeaveFourDiffOut():

    def __init__(self,
                 model,
                 path_0_original = "images/Anciennes_cartes/",
                 path_1_original= "images/Nouvelles_cartes/",
                 path_0_keras = "images/Anciennes_cartes/Keras_photos/",
                 path_1_keras = "images/Nouvelles_cartes/Keras_photos/",
                 pickle_original="data.pickle",
                 pickle_keras="new_data.pickle"):
        self.model = model
        self.path_0_original = path_0_original
        self.path_1_original = path_1_original
        self.path_0_keras = path_0_keras
        self.path_1_keras = path_1_keras
        self.pickle_original = pickle_original
        self.pickle_keras = pickle_keras
        self.data = []


    def data_(self):
        colors_data_original = ColorsData(cat_paths=[self.path_0_original, self.path_1_original], data_file_name=self.pickle_original)
        colors_data_original.get_data()
        data_original = colors_data_original.data

        colors_data_keras = ColorsData(cat_paths=[self.path_0_keras, self.path_1_keras], data_file_name=self.pickle_keras)
        colors_data_keras.get_data()
        data_keras = colors_data_keras.data

        self.data = [data_original, data_keras]
        return self.data


    def split(self):
        if len(self.data) != 2:
            self.data = data_
        data_original = self.data[0]
        data_keras = self.data[1]
        sizes = []



        return 0