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
            self.data = self.data_()
        data_original = self.data[0]
        data_keras = self.data[1]
        sizes = []

        X_original, y_original = [donnee[0] for donnee in data_original], [donnee[1] for donnee in data_original]
        X_keras, y_keras = [donnee[0] for donnee in data_keras], [donnee[1] for donnee in data_keras]

        n_splits = min([data_original[1].count(0),data_original[1].count(1)])
        splits = {}
        for i in range(n_splits):
            splits[i] = {}
            # splits[i]["X_train"] = np.array(X)[train_index]
            splits[i]["X_test"] = []
            splits[i]["X_test"].append()
            # splits[i]["y_train"] = np.array(y)[train_index]
            # splits[i]["y_test"] = np.array(y)[test_index]

        return 0


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    lfdo=LeaveFourDiffOut(model=LogisticRegression())
    lfdo.data_()
    print(lfdo.split())
    # [[[205, 192, 214], 0], [[177, 165, 173], 0], [[183, 180, 189], 0], [[195, 179, 182], 0], [[191, 180, 178], 0], [[142, 130, 138], 0], [[213, 200, 208], 0], [[206, 198, 210], 0], [[124, 149, 166], 0], [[159, 116, 110], 0], [[191, 181, 198], 0], [[202, 185, 181], 0], [[179, 169, 185], 0], [[125, 114, 112], 0], [[198, 187, 190], 0], [[150, 173, 182], 1], [[175, 173, 169], 1], [[158, 149, 139], 1], [[209, 200, 196], 1], [[148, 144, 140], 1], [[167, 166, 161], 1], [[143, 138, 130], 1], [[187, 194, 192], 1], [[123, 113, 108], 1], [[182, 190, 197], 1], [[133, 120, 104], 1], [[160, 153, 141], 1]]
