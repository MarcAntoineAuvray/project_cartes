import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class ColorsModel:

    def __init__(self, model, cat_names=[0,1]):
        self.model = model
        self.cat_names = cat_names

    def add_new_cat_names(self, new_cat_names):
        self.cat_names=new_cat_names
        return self.cat_names

    def fit(self, colors_data):
        data = colors_data
        X_train, y_train = data
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        if len(np.shape(X)) == 1:
            X = [X]
        y_pred = self.model.predict(X)
        return y_pred

    def fit_predict(self, colors_data, new_cat_names=None):
        data = colors_data
        self.fit(data)
        X, y = data
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



if __name__ == "__main__":
    ex_colors_model = ColorsModel(model=LogisticRegression())

    # for df_ in ex_colors_model.test_model(): print(df_["comparaison"].value_counts())

#cnn réseau de neuronnes convolutionnel
#pour augmenter le nb de données pour pouvoir faire le cnn on doit faire de la data augmentation (adversial network) GAN