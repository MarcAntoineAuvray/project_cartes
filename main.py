from ArrayData import ArrayData
from CNNModel import CNNModel

# troisieme modele 27 puis 297 images :
# modele : CNN
# donnees : total des info des images (reduites a 32x32) + donnees creees par la data augmentation
arr_ = ArrayData(cat_paths=["images/Anciennes_cartes/Keras_photos/",
                            "images/Nouvelles_cartes/Keras_photos/"])
arr_.get_data_and_target()
cnn_ = CNNModel(data=arr_.data, target=arr_.target)
cnn_.fit(do_split=True, n_test=30, verbose=True, new_cat_names=["old", "new"])

# passage du main.py vers un main_notebook.ipynb

# 07 01 2023
# 14 21
# contenu du word legerement ameliore
# finito v2