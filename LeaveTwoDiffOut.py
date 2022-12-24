import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt


class LeaveTwoDiffOut():

    def __init__(self, model, data, target, cat_names=[0, 1]):
        self.model = model
        self.data = data
        self.target = target
        self.cat_names = cat_names
        self.splits = {}


    def split(self):
        X = np.array(self.data)
        y = np.array(self.target)

        splits = {}

        leave_p_out = sklearn.model_selection.LeavePOut(p=2)
        leave_p_out.get_n_splits(X)
        i = 0
        for train_index, test_index in leave_p_out.split(X):
            for x in range(0, len(test_index)):
                for z in range(1, len(test_index)):
                    if (y[test_index[x]] != y[test_index[z]]):
                        splits[i] = {}
                        splits[i]["X_train"] = np.array(X)[train_index]
                        splits[i]["X_test"] = np.array(X)[test_index]
                        splits[i]["y_train"] = np.array(y)[train_index]
                        splits[i]["y_test"] = np.array(y)[test_index]
                        self.model.fit(data=splits[i]["X_train"], target=splits[i]["y_train"])
                        splits[i]["y_test_pred"] = self.model.predict(data=splits[i]["X_test"])
                        i = i + 1
        self.splits = splits
        return self.splits

    def trues_falses(self):
        if len(self.splits) == 0:
            self.split()
        splits = self.splits
        n_trues = 0
        n_falses = 0
        for i in range(len(splits)):
            for j in range(len(splits[i]["y_test"])):
                n_trues = n_trues + list(splits[i]["y_test"] == splits[i]["y_test_pred"]).count(True)
                n_falses = n_falses + list(splits[i]["y_test"] == splits[i]["y_test_pred"]).count(False)
        return n_trues, n_falses

    def description(self):
        n_trues, n_falses = self.trues_falses()
        print("Sur", int((n_trues+n_falses)/2), "modeles avec des jeux d'entrainements differents.",
              "Il y a eu",
              "".join([str(round(100 * n_trues / (n_trues + n_falses), 2)), "% de bonnes predictions sur le jeu de test."]),
              "".join([str(round(100 * n_falses / (n_trues + n_falses), 2)), "% de mauvaises predictions sur le jeu de test."]))

    def pie(self):
        n_trues, n_falses = self.trues_falses()
        plt.pie(x=[n_trues, n_falses],
                colors=["green", "red"],
                labels=["".join(["true : ", str(round(100 * n_trues / (n_trues + n_falses), 2)), "%"]),
                        "".join(["false : ", str(round(100 * n_falses / (n_trues + n_falses), 2)), "%"])],
                labeldistance=None)
        plt.title("% of true or false predictions (in test data)")
        plt.legend()
        plt.show()




