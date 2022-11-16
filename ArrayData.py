from PIL import Image
import os
import numpy as np
import pickle
import json
import statistics as stat
from PIL import Image
from numpy import asarray

class ArrayData:
    def __init__(self,
                 files_path="C:/Users/mauvray/PycharmProjects/cartes/",
                 cat_paths=["images/Anciennes_cartes/","images/Nouvelles_cartes/"],
                 data_file_name="data.pickle",
                 cat_names=["old_cards", "new_cards"],
                 new_sizes=(32,32)):

        self.files_path = files_path
        self.cat_paths = cat_paths
        self.data_file_name = data_file_name
        self.cat_names = cat_names
        self.data=[]
        self.new_sizes=new_sizes

    def data_(self):
        X = []
        y = []
        for i in [0, 1]:
            for file in [file_ for file_ in os.listdir(self.files_path + self.cat_paths[i]) if
                         file_[-4:] != ".png" and file_ != "Keras_photos"]:
                X.append(asarray(Image.open(self.files_path + self.cat_paths[i] + file).resize(size=self.new_sizes)) / 255)
                y.append(i)
        return X, y


if __name__ == "__main__":
    arr_=ArrayData(cat_paths=["images/Anciennes_cartes/Keras_photos/","images/Nouvelles_cartes/Keras_photos/"])
    from CNNModel import CNNModel
    cnn_=CNNModel()
    cnn_.fit(arr_.data_())