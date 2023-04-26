import os
from keras.models import load_model

current_path = os.path.dirname(os.path.realpath(__file__))
RAKURSES = ['0', '35', '90', '145', '180', '215', '270', '325']


def model_load(path: str, model: str):
    """
    Принимает на вход 2 параметра:
    path - путь к файлам модели (path like obj)
    model - название файла модели (str типа M_good190.h5)
    На выход возвращает собранную модель
    """
    g_model = load_model(os.path.join(path, 'data', 'model', model), compile=True)
    return g_model

