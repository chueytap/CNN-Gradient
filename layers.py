

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import models, layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import models
from tensorflow.keras import layers


def build_resnet_model(input_shape=(128, 128, 3)):
    """
    Builds a ResNet50-based model for image classification.

    Args:
        input_shape (tuple): The shape of the input images. Defaults to (128, 128, 3).

    Returns:
        A Keras Sequential model.
    """
    conv_base = ResNet50(weights='imagenet',
                         include_top=False, input_shape=input_shape)

    for layer in conv_base.layers:
        layer.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(14, activation='softmax'))

    return model
