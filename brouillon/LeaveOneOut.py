import numpy as np
import sklearn.model_selection
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt


class LeaveOneOut():

    def __init__(self, model, cat_names=[0,1]):
        self.model = model
        self.cat_names = cat_names

    def split(self, data):
        X, y = data
        leave_one_out = sklearn.model_selection.LeaveOneOut()
        leave_one_out.get_n_splits(X)
        splits={}
        i=0
        for train_index, test_index in leave_one_out.split(X):
            splits[i]={}
            splits[i]["X_train"] = np.array(X)[train_index]
            splits[i]["X_test"] = np.array(X)[test_index]
            splits[i]["y_train"] = np.array(y)[train_index]
            splits[i]["y_test"] = np.array(y)[test_index]
            self.model.fit(data=(splits[i]["X_train"], splits[i]["y_train"]))
            splits[i]["y_test_pred"]=self.model.predict(splits[i]["X_test"] )
            i=i+1

        return splits

    def pie(self, data):
        splits=self.split(data)
        true=0
        false=0
        for i in range(len(splits)):
            if splits[i]["y_test"]==splits[i]["y_test_pred"]:
                true=true+1
            if splits[i]["y_test"]!=splits[i]["y_test_pred"]:
                print(splits[i]["y_test"])
                print(splits[i]["y_test_pred"])
                false=false+1

        plt.pie(x=[true,false],
                colors=["green","red"],
                labels=["".join(["true : ", str(round(100*true/(true+false),2)),"%"]),
                        "".join(["false : ", str(round(100*false/(true+false),2)),"%"])],
                labeldistance =None)
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
    loo_ = LeaveOneOut(model=colors_model)
    print(loo_.split(colors_data))
    loo_.pie(colors_data)
