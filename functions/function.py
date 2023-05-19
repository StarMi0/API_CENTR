import os
import cv2
import keras as keras
import numpy as np


def insert_array(original_array, insert_array):
    original_shape = np.shape(original_array)
    insert_shape = np.shape(insert_array)
    row_diff = int((original_shape[0] - insert_shape[0]) / 2)
    col_diff = int((original_shape[1] - insert_shape[1]) / 2)
    output_array = np.zeros(original_shape, dtype=int)
    output_array[row_diff:(row_diff + insert_shape[0]), col_diff:(col_diff + insert_shape[1])] = insert_array

    return output_array


def read_img(dir) -> np.array:
    """
    his function reads in a single image from the given directory, crops it to the desired size, and then converts it
    to a numpy array with binary values (-1 for 0 and 1 for non-zero pixels). The output is a numpy array with shape
    (2048, 2048, 1).
    """
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE).astype(np.bool_).astype(np.int8)
    new_array = np.zeros((1024, 1024))
    img = insert_array(img, new_array)
    img[img == 0] = -1
    return np.expand_dims(img, axis=2)  # shape np.array (2048, 2048, 1)


def get_img_for_predict(dir_folder: os.PathLike) -> np.array:
    """
    This function is used to read in the images in the given directory, sort them, and then concatenate them into a
    single array. The output is a numpy array with shape (1, 2048, 2048, 8).
    """
    list_img = []
    files = os.listdir(dir_folder)
    files.sort()
    for filename in files:
        if filename.endswith('.png'):
            list_img.append(read_img(os.path.join(dir_folder, filename)))

    X1 = np.concatenate(list_img, axis=-1)
    return np.expand_dims(X1, axis=0)  # shape np.array (1, 2048, 2048, 8)


def predict_img(img: np.array, model: keras.models) -> np.array:
    """
    This function loads the given model and uses it to predict the pixel values of the given image. The model output
    is then scaled and shifted so that it is in the range of 0-255, and any values less than 128 are replaced with 0.
    The output is a numpy array with pixel values in the range of 0-255.
    """
    g_model = input_data.model_load(input_data.current_path, model)
    img = g_model.predict(img)[0] * 127.5 + 127.5
    return np.where(img < 128, 0, 255)  # shape np.array (2048, 2048, 8)


def save_gen_img(img, path):
    """
    This function takes an input image and a path, and then saves each of the eight slices of the image as an
    individual .png file in the given path. The output is eight separate files containing the data from the input
    image.
    """
    Z = np.zeros((8, 1344, 1008))
    for i in range(8):
        Z[i, :, :] = img[:, :1008, i]
        cv2.imwrite(os.path.join(path, f'{str(i)}.png'), Z[i])
