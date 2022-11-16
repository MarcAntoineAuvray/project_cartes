from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LogisticRegression

from ColorsModel import ColorsModel
from ColorsData import ColorsData

ex_colors_model = ColorsModel(model=LogisticRegression())
data_class = ColorsData()

ex_colors_model.fit_predict(data=data_class.get_data(),
                            cat_names=data_class.cat_names)
ex_colors_model.split_data(data_class.get_data())



