from PIL import Image
import os
import numpy as np
import pickle
import statistics as stat

class ColorsData:
    def __init__(self,
                 files_path="C:/Users/mauvray/PycharmProjects/cartes/",
                 cat_paths=["images/Anciennes_cartes/","images/Nouvelles_cartes/"],
                 data_file_name="data.pickle",
                 cat_names=["old_cards", "new_cards"]):

        self.files_path = files_path
        self.cat_paths = cat_paths
        self.data_file_name = data_file_name
        self.cat_names = cat_names
        self.data = []

    def data_X(self, image_file, path):
        img = Image.open(path + image_file)
        img.convert('RGB')
        list_lists = []
        for w in range(img.size[0]):
            for h in range(img.size[1]):
                red, green, blue = img.getpixel((w, h))
                list_lists.append([red, green, blue])
        X = [stat.mean(list_) for list_ in np.array(list_lists).transpose()]
        return X

    def data_X_y(self, image_file, one_path):
        X = self.data_X(image_file, path=self.files_path+one_path)
        y = self.cat_paths.index(one_path)
        return X, y

    def add_image_to_data(self, image_file, one_path, update_all=False):
        X, y = self.data_X_y(image_file, one_path)
        self.data.append([X, y])
        return self.data

    def get_data(self, update_data=False):
        if update_data:
            self.data = []
            for i in [0, 1]:
                for file in [file_ for file_ in os.listdir(self.cat_paths[i]) if file_[-4:] != ".png"]:
                    self.add_image_to_data(file, self.cat_paths[i])
            with open(self.data_file_name, 'wb') as f:
                pickle.dump(self.data, f)
            X = [row for row in np.transpose(self.data)[0]]
            y = list(np.transpose(self.data)[1])
            return X, y
        else:
            with open("data.pickle", "rb") as openfile:
                self.data = pickle.load(openfile)
            X = [row for row in np.transpose(self.data)[0]]
            y = list(np.transpose(self.data)[1])
            return X, y
if __name__ == "__main__":
    data_ = ColorsData()
    for i in [0, 1]:
        for file in [file_ for file_ in os.listdir(data_.cat_paths[i]) if file_[-4:] != ".png"]:
            data_.add_image_to_data([file], data_.cat_paths[i])
    print(data_.read_data())
