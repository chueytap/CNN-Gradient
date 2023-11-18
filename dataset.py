import os
import numpy as np
from PIL import Image
from keras.utils import np_utils


def load_dataset(dataset_path, input_shape=(128, 128), split_ratio=0.7):
    """
    Loads a dataset from the given path and returns the training and testing data.

    Args:
        dataset_path (str): The path to the dataset directory.
        input_shape (tuple, optional): The desired shape of the input images. Defaults to (128, 128).
        split_ratio (float, optional): The ratio of training data to testing data. Defaults to 0.7.

    Returns:
        tuple: A tuple containing the training and testing data, each as a tuple of numpy arrays.
    """
    x_train = []
    y_train = []
    class_folders = os.listdir(dataset_path)
    num_classes = len(class_folders)
    print('Number of classes: {}'.format(num_classes))

    for class_id, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_folder)
        image_files = os.listdir(class_path)
        print('Class: {}, number of images: {}'.format(
            class_folder, len(image_files)))

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path)
            image = image.resize(input_shape, Image.LANCZOS)
            image = np.array(image)

            x_train.append(image)
            y_train.append(class_id)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    permutation = np.random.permutation(len(x_train))
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    split_index = int(len(x_train) * split_ratio)
    x_test = x_train[split_index:]
    y_test = y_train[split_index:]
    x_train = x_train[:split_index]
    y_train = y_train[:split_index]

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('x_test shape: {}'.format(x_test.shape))
    print('y_test shape: {}'.format(y_test.shape))

    return (x_train, y_train), (x_test, y_test)
