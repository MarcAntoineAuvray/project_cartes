from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytesseract import pytesseract
import json

files_path="C:/Users/mauvray/PycharmProjects/cartes/images/"

path_to_tesseract = "C://Program Files//Tesseract-OCR//tesseract.exe"

list_words_new_cards=["titre","séjour","sejour",
                      "noms","prénoms","prenoms",
                      "surnames","forenames",
                      "sexe","sex"
                      "nationalité","nationalite","nat."
                      "date","naissance","birth",
                      "cat.","type","permit",
                      "valable","jusqu","valid","until"
                      "numero","personnel","personal","number"
                      "residence","permit"]

list_words_old_cards = ["titre", "séjour", "sejour",
                        "carte", "resident",
                        "nom", "prénom", "prenom",
                        "validité", "validite", "debut",
                        "delivre", "délivré",
                        "motif",
                        "signature", "aurorité", "autorite", "lautorite"]

dict_words_from_cards = {"old_cards": list_words_old_cards, "new_cards": list_words_new_cards}


def recup_files_frompath(files_path):
    list_files = []
    for path in os.listdir(files_path):
        if os.path.isfile(os.path.join(files_path, path)):
            list_files.append(path)
    return list_files


def image_to_pil(image_file):
    format=image_file[image_file.rfind(".")+1:]
    list_accepted_formats=["jpg","jpeg","png"]
    if format in list_accepted_formats:
        img = Image.open(image_file)
        return img
    else:
        img = Image.open(image_file)
        return img

def show_pil(pil):
    return pil.show()

def pil_to_str(pil):
    path_to_tesseract = "C://Program Files//Tesseract-OCR//tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    pil=pytesseract.image_to_string(pil).lower()
    return pil

def image_to_text(pil):
    return pil_to_str(pil)

def clean_text(str):
    str = str.replace('/', '')
    str = str.replace('=', '')
    str = str.replace('<', '')
    str = str.replace('\n', ' ')
    return str

def rotate_pil(pil,number_rotations):
    return pil.rotate(number_rotations)

def rotate_text(pil, number_rotations):
    return pil_to_str(rotate_pil(pil,number_rotations))

def count_words_fromcards_inourtxt(txt, dict_words_from_cards):
    list_words_both = list(set(dict_words_from_cards["old_cards"]) | set(dict_words_from_cards["new_cards"]))
    number_of_trues=0
    for mot in clean_text(txt).split():
        if mot in list_words_both:
            number_of_trues=number_of_trues+1
    return number_of_trues

def find_best_rotations(pil):
    list_numbers_words=[]
    list_i=[]
    list_texts=[]
    i=0
    while i <360:
        pil_rotated=rotate_pil(pil,i)
        txt=pil_to_str(pil_rotated)
        number=count_words_fromcards_inourtxt(txt,dict_words_from_cards)
        txt=clean_text(txt)
        if number == 0:
            list_texts.append(txt)
            list_numbers_words.append(number)
            list_i.append(i)
            i=i+6
        else:
            list_texts.append(txt)
            list_numbers_words.append(number)
            list_i.append(i)
            i=i+1

    return (list_texts,list_numbers_words,list_i)


def best_text(pil):
    list_lists=find_best_rotations(pil)

    list_texts=list_lists[0]
    list_numbers_words=list_lists[1]
    list_i=list_lists[2]

    index_max_number=list_numbers_words.index(max(list_numbers_words))

    return list_texts[index_max_number]

def save_as_json(path_old_cards, path_new_cards, dict_words_from_cards):
    list_texts_founded_old_cards = list([ best_text(image_to_pil(path_old_cards+file_name)) for file_name in  recup_files_frompath(path_old_cards)])
    list_texts_founded_new_cards = list([ best_text(image_to_pil(path_new_cards+file_name)) for file_name in  recup_files_frompath(path_new_cards)])

    dict_info_= {"old_cards" : {"words_expected" : dict_words_from_cards["old_cards"], "words_found" : list_texts_founded_old_cards },
                 "new_cards" : {"words_expected" : dict_words_from_cards["new_cards"], "words_found" : list_texts_founded_new_cards }}
    with open('words_expected_and_words_founded.json', 'w') as f:
        json.dump(dict_info_, f)

def read_json():
    with open('words_expected_and_words_founded.json', 'r') as f:
        return json.load(f)

