import numpy as np
import sklearn.model_selection
from sklearn.model_selection import LeavePOut
import matplotlib.pyplot as plt


class LeaveTwoDiffOut():

    def __init__(self, model, cat_names=[0,1]):
        self.model = model
        self.cat_names = cat_names
        self.dict = {}

    def split(self, colors_data):
        data = colors_data
        X, y = data
        leave_p_out = sklearn.model_selection.LeavePOut(p=2)
        leave_p_out.get_n_splits(X)
        splits={}
        i=0
        for train_index, test_index in leave_p_out.split(X):
            for x in range(0, len(test_index)):
                for z in range(1, len(test_index)):
                    if (y[test_index[x]] != y[test_index[z]]):
                        splits[i]={}
                        splits[i]["X_train"] = np.array(X)[train_index]
                        splits[i]["X_test"] = np.array(X)[test_index]
                        splits[i]["y_train"] = np.array(y)[train_index]
                        splits[i]["y_test"] = np.array(y)[test_index]
                        self.model.fit(colors_data=(splits[i]["X_train"], splits[i]["y_train"]))
                        splits[i]["y_test_pred"]=self.model.predict(splits[i]["X_test"] )
                        i=i+1
        self.dict = splits
        return self.dict

    def pie(self, colors_data):
        data = colors_data
        splits=self.split(data)
        true=0
        false=0
        for i in range(len(splits)):
            for j in range(len(splits[i]["y_test"])):
                if splits[i]["y_test"][j]==splits[i]["y_test_pred"][j]:
                    true=true+1
                if splits[i]["y_test"][j]!=splits[i]["y_test_pred"][j]:
                    false=false+1
        plt.pie(x=[true, false],
                colors=["green", "red"],
                labels=["".join(["true : ", str(round(100 * true / (true + false), 2)), "%"]),
                        "".join(["false : ", str(round(100 * false / (true + false), 2)), "%"])],
                labeldistance=None)
        plt.title("% of true or false predictions (in test data)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    from ColorsData import *
    from ColorsModel import *
    from sklearn.linear_model import LogisticRegression
    colors_data = ColorsData().get_data()
    print(colors_data)
    colors_model= ColorsModel(model=LogisticRegression())
    lpo = LeaveTwoDiffOut(model=colors_model)
    print(lpo.split(colors_data))
    lpo.pie(colors_data)
