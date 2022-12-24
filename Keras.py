from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

import os

files_path="C:/Users/mauvray/PycharmProjects/cartes/"
cat_paths=["images/Anciennes_cartes/","images/Nouvelles_cartes/"]
keras_photos_path="Keras_photos/"


def one_data_augmentation(image_file,
                          cat,
                          number_photos=10,
                          files_path="C:/Users/mauvray/PycharmProjects/cartes/",
                          cat_paths=["images/Anciennes_cartes/","images/Nouvelles_cartes/"],
                          keras_photos_path="Keras_photos/"):
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    img = load_img(files_path+cat_paths[cat]+image_file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x,
                              batch_size=1,
                              save_to_dir=files_path+cat_paths[cat]+keras_photos_path,
                              save_prefix=image_file,
                              save_format='jpeg'):
        i += 1
        if i > number_photos:
            break

def loop_data_augmentation():
    for cat in [0,1]:
        for image_file in os.listdir(files_path+cat_paths[cat]):
            if image_file[-4:] != ".png" and image_file != keras_photos_path[:-1]:
                one_data_augmentation(image_file=image_file, cat=cat)

