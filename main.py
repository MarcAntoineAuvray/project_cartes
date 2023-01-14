from ArrayData import ArrayData
from CNNModel import CNNModel

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

import os

import numpy as np
import pandas as pd
import random
from scikeras.wrappers import KerasClassifier

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# troisieme modele 27 puis 297 images :
# modele : CNN
# donnees : total des info des images (reduites a 32x32) + donnees creees par la data augmentation
arr_ = ArrayData(cat_paths=["images/Anciennes_cartes/Keras_photos/",
                            "images/Nouvelles_cartes/Keras_photos/"])
arr_.get_data_and_target()
cnn_ = CNNModel(data=arr_.data, target=arr_.target)
# cnn_.fit(do_split=True, n_test=3, verbose=True, new_cat_names=["old", "new"])

# passage du main.py vers un main_notebook.ipynb

# 14 01 2023
# 15 h 01
# pdf rapport