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


class CNNModel:

    def __init__(self, data, target):
        self.data = data
        self.dim_1, self.dim_2, self.dim_3, self.dim_4 = np.shape(data)
        self.target = target
        self.X_y_prepared = False
        self.sets = {}
        self.sets_detailled = pd.DataFrame()
        self.pipeline = None
        self.cvs = None

    def prepare_X_y(self):
        X = np.array(self.data).reshape(self.dim_1, self.dim_4 * self.dim_2 * self.dim_3).astype(float)
        encoder = LabelEncoder()
        encoder.fit(self.target)
        y = encoder.transform(self.target)
        self.X_y_prepared = True
        return X, y

    def split(self, n_test=2):
        X, y = self.prepare_X_y()
        test_index = random.sample(range(self.dim_1), n_test)
        train_index = [i for i in range(self.dim_1) if i not in test_index]
        sets = {"X_train": np.array(X)[train_index], "X_test": np.array(X)[test_index],
                "y_train": np.array(y)[train_index], "y_test": np.array(y)[test_index]}
        self.sets = sets

    def fit(self, do_split=False, n_test=2, verbose=False, new_cat_names=None):
        if do_split:
            self.split(n_test)

        else:
            X, y = self.prepare_X_y()
            self.sets = {"X_train": np.array(X), "X_test": np.array([]),
                         "y_train": np.array(y), "y_test": np.array([])}

        model_ = Sequential()
        model_.add(Dense(self.dim_4 * self.dim_2 * self.dim_3,
                         input_shape=(self.dim_4 * self.dim_2 * self.dim_3,),
                         activation='relu'))
        model_.add(Dense((self.dim_4 * self.dim_2 * self.dim_3)/2, activation='relu'))
        model_.add(Dense(1, activation='sigmoid'))
        model_.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        keras_classifier = KerasClassifier(model=model_, epochs=10, batch_size=5, verbose=0)

        pipeline = Pipeline([('standardize', StandardScaler()), ('mlp', keras_classifier)])
        pipeline.fit(self.sets["X_train"], self.sets["y_train"])
        self.pipeline = pipeline
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        self.cvs = cross_val_score(pipeline, self.sets["X_train"], self.sets["y_train"], cv=kfold)

        if do_split:
            self.sets["y_test_pred"] = np.array(pipeline.predict(self.sets["X_test"]))
            self.sets_detailled = pd.DataFrame({"y_test": self.sets["y_test"],
                                                "y_test_pred": self.sets["y_test"],
                                                "proba_0": [round(proba[0], 3) for proba in self.pipeline.predict_proba(self.sets["X_test"])],
                                                "proba_1": [round(proba[1], 3) for proba in self.pipeline.predict_proba(self.sets["X_test"])],
                                                "comparaison_observed_predicted": [self.sets["y_test_pred"][i] == self.sets["y_test"][i] for i in range(n_test)]})
            if new_cat_names:
                self.sets_detailled["y_names"] = [new_cat_names[value] for value in self.sets_detailled["y_test"]]
                self.sets_detailled["y_pred_names"] = [new_cat_names[value] for value in self.sets_detailled["y_test_pred"]]

        if verbose:
            print("Qualite : %.2f%% (%.2f%%)" % (self.cvs.mean() * 100, self.cvs.std() * 100))
            if do_split:
                print("number of images to test:", n_test)
                print("good (True) or bad (False) predictions :\n",
                      self.sets_detailled["comparaison_observed_predicted"].value_counts())


