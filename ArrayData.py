from numpy import asarray
import os
from PIL import Image

class ArrayData:
    def __init__(self,
                 files_path="C:/Users/mauvray/PycharmProjects/cartes/",
                 cat_paths=["images/Anciennes_cartes/","images/Nouvelles_cartes/"],
                 new_sizes=(32, 32)):

        self.files_path = files_path
        self.cat_paths = cat_paths
        self.new_sizes = new_sizes
        self.data = []
        self.target = []

    def get_data_and_target(self):
        X = []
        y = []
        for i in [0, 1]:
            for file in [file_ for file_ in os.listdir(self.files_path + self.cat_paths[i]) if
                         file_[-4:] != ".png" and file_ != "Keras_photos"]:
                X.append(asarray(Image.open(self.files_path + self.cat_paths[i] + file).resize(size=self.new_sizes)) / 255)
                y.append(i)

        self.data = X
        self.target = y
        return self.data, self.target

