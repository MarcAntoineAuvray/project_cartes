import numpy as np
import pandas as pd

class ColorsModel:

    def __init__(self, model, cat_names=[0,1]):
        self.model = model
        self.cat_names = cat_names

    def add_new_cat_names(self, new_cat_names):
        self.cat_names=new_cat_names
        return self.cat_names

    def fit(self, data, target):
        X_train, y_train = (data, target)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, data):
        X=data
        if len(np.shape(X)) == 1:
            X = [X]
        y_pred = self.model.predict(X)
        return y_pred

    def fit_predict(self, data, target, new_cat_names=None):
        self.fit(data=data, target=target)
        X, y = (data, target)
        y_pred = self.predict(X)
        if new_cat_names :
            self.add_new_cat_names(new_cat_names)
            df = pd.DataFrame({'y': y,
                               'y_pred': y_pred,
                               "y_names": [self.cat_names[value] for value in y],
                               "y_pred_names": [self.cat_names[value] for value in y_pred],
                               "comparaison" : [ y[i]==y_pred[i] for i in range(len(y))]})
            print(df)
            return df
        else:
            return y_pred

