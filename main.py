from ColorsModel import ColorsModel
from ColorsData import ColorsData
from CNNModel import CNNModel
from ArrayData import ArrayData
from Keras import *
from LeaveTwoDiffOut import LeaveTwoDiffOut

import lazypredict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lazypredict.Supervised import LazyClassifier
import pandas as pd

# premier modele avec 27 images :
#  modele : regression logistique
#  donnee des images : moyennes
#  methode pour valider : enlever une image de chacune des 2 cat pour trainer sur le reste, tester sur les 2
data_class = ColorsData()
data_class.get_data_and_target()

l2o = LeaveTwoDiffOut(model=ColorsModel(model=LogisticRegression()))
l2o.data = data_class.data
l2o.target = data_class.target
l2o.split()
l2o.pie()
l2o.description()

# deuxieme modele avec 27 images :
# modele.s : plusieurs modeles fournis par le package lazymodel
# donnee.s des images : moyenne, mediane
# methode pour valider : indicateurs fournis par le package
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None)

list_models = []
for i in range(len(l2o.splits)):
    X_train, X_test, y_train, y_test, y_test_pred = l2o.splits[0].values()
    models, predictions = clf.fit(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))
    list_models.append(models)

df = pd.concat(list_models)
print("Nombre de types modeles differents:",
      len(df)/(i+1))
print("Nombre de splits :",
      i+1)
print("Nombre total de modeles differents :",
      len(df))


# 30 11 2022
# 17 : 28
# ajout du notebook



