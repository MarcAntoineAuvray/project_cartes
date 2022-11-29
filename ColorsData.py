from PIL import Image
import os
import numpy as np
import pickle
import statistics as stat

class ColorsData:
    def __init__(self,
                 files_path="C:/Users/mauvray/PycharmProjects/cartes_/project_cartes/",
                 cat_paths=["images/Anciennes_cartes/","images/Nouvelles_cartes/"],
                 data_file_name="data.pickle",
                 cat_names=["old_cards", "new_cards"]):

        self.files_path = files_path
        self.cat_paths = cat_paths
        self.data_file_name = data_file_name
        self.cat_names = cat_names
        self.data = []
        self.target = []

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

    def add_image_to_data(self, image_file, one_path):
        X = self.data
        y = self.target
        new_X, new_y = self.data_X_y(image_file, one_path)
        X.append(new_X)
        y.append(new_y)
        self.data = X
        self.target = y
        return self.data, self.target

    def get_data_and_target(self, update=False, updated_file_name="data.pickle"):
        if update:
            self.data = []
            self.target = []
            for i in [0, 1]:
                for file in [file_ for file_ in os.listdir(self.cat_paths[i]) if file_[-4:] != ".png" and file_ != "Keras_photos"]:
                    self.add_image_to_data(image_file=file, one_path=self.cat_paths[i])
            with open(updated_file_name, 'wb') as f:
                pickle.dump((self.data, self.target), f)

            X = self.data
            y = list(self.target)
            self.data = X
            self.target = y
            return self.data, self.target
        else:
            with open(self.data_file_name, "rb") as openfile:
                self.data, self.target = pickle.load(openfile)
            X = self.data
            y = list(self.target)
            self.data = X
            self.target = y
            return self.data, self.target

if __name__ == "__main__":
    data_ = ColorsData()
    # for i in [0, 1]:
    #     for file in [file_ for file_ in os.listdir(data_.cat_paths[i]) if file_[-4:] != ".png"]:
    #         data_.add_image_to_data(file, data_.cat_paths[i])
    print(data_.get_data_and_target())

    # c__ = ColorsData(cat_paths=["images/Anciennes_cartes/Keras_photos/",
    #                             "images/Nouvelles_cartes/Keras_photos/"]).get_data(update_data=True,
    #                                                                                updated_data_file_name="new_data.pickle")
