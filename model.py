
import os
from layers import build_resnet_model
from tensorflow.keras import optimizers, metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras import optimizers
from tensorflow.keras import metrics
from imblearn.over_sampling import SMOTE

from tensorflow.keras import optimizers

from support import F1ScoreCallback


def resnet_model(x_train, y_train, x_test, y_test):
    """
    Trains a ResNet model on the given training data and evaluates it on the given test data.

    Args:
        x_train (numpy.ndarray): The training input data.
        y_train (numpy.ndarray): The training target data.
        x_test (numpy.ndarray): The test input data.
        y_test (numpy.ndarray): The test target data.

    Returns:
        None
    """
    model = build_resnet_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=int(
        os.getenv('EARLY_STOPPING_PATIENCE', 100)))

    model.compile(
        optimizer=optimizers.RMSprop(

            learning_rate=float(os.getenv('LEARNING_RATE', 2e-5))),
        loss='categorical_crossentropy',
        metrics=[
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
        ]
    )
    history = model.fit(x_train, y_train, epochs=int(os.getenv('EPOCHS', 50)), batch_size=int(os.getenv('BATCH_SIZE', 8)),
                        validation_data=(x_test, y_test),
                        callbacks=[F1ScoreCallback(), early_stopping])

    model_json = model.to_json()
    with open(os.getenv('MODEL_JSON_PATH', "./models/resnet.json"), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.getenv('MODEL_WEIGHTS_PATH', "./models/resnet.h5"))

    # Plot the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.getenv('PLOT_PATH', "./graphs/resnet.png"))


def smote_model(x_test, y_test, x_train, y_train):
    """
    Trains a ResNet model using SMOTE oversampling on the training data.

    Args:
        x_test (numpy.ndarray): The test data.
        y_test (numpy.ndarray): The test labels.
        x_train (numpy.ndarray): The training data.
        y_train (numpy.ndarray): The training labels.

    Returns:
        None
    """
    y_train_mc = np.argmax(y_train, axis=1)

    X_train_2d = x_train.reshape((x_train.shape[0], -1))

    sm = SMOTE(random_state=42)
    x_train_res, y_train_res = sm.fit_resample(X_train_2d, y_train_mc)

    x_train_res = x_train_res.reshape(
        (-1, x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    y_train_res = np_utils.to_categorical(y_train_res, 14)

    model = build_resnet_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=int(
        os.getenv('EARLY_STOPPING_PATIENCE', 100)))

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=float(
            os.getenv('SMOTE_LEARNING_RATE', 2e-5))),
        loss='categorical_crossentropy',
        metrics=[
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
        ]
    )
    history = model.fit(x_train_res, y_train_res, epochs=int(os.getenv('SMOTE_EPOCHS', 50)),
                        batch_size=int(os.getenv('SMOTE_BATCH_SIZE', 8)),
                        validation_data=(x_test, y_test), callbacks=[F1ScoreCallback(), early_stopping])

    model_json = model.to_json()
    with open(os.getenv('SMOTE_MODEL_JSON_PATH', "./models/smote.json"), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(
        os.getenv('SMOTE_MODEL_WEIGHTS_PATH', "./models/smote.h5"))

    # Plot the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.getenv('SMOTE_PLOT_PATH', "./graphs/smote.png"))


def gradient_model(x_train, y_train, x_test, y_test):
    """
    Trains a CatBoostClassifier model on the given training data and returns the trained model.

    Args:
    x_train (numpy.ndarray): Training data features.
    y_train (numpy.ndarray): Training data labels.
    x_test (numpy.ndarray): Test data features.
    y_test (numpy.ndarray): Test data labels.

    Returns:
    CatBoostClassifier: Trained CatBoostClassifier model.
    """

    y_train_mc = np.argmax(y_train, axis=1)
    y_test_mc = np.argmax(y_test, axis=1)

    model = CatBoostClassifier(iterations=int(os.getenv('GRADIENT_ITERATIONS', 300)),
                               learning_rate=float(
                                   os.getenv('GRADIENT_LEARNING_RATE', 0.1)),
                               depth=int(os.getenv('GRADIENT_DEPTH', 12)),
                               loss_function='MultiClass')
    model.fit(x_train, y_train_mc, eval_set=(
        x_test, y_test_mc), early_stopping_rounds=100)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test_mc, y_pred)
    print("Accuracy: ", accuracy)

    report = classification_report(y_test_mc, y_pred)
    print(report)

    matrix = confusion_matrix(y_test_mc, y_pred)
    print("Confusion Matrix: \n", matrix)

    pickle.dump(model, open(os.getenv('GRADIENT_MODEL_PATH',
                "./models/gradient.pickle.dat"), "wb"))

    print("Saved model to disk")

    return model


def svm_model(x_train, y_train, x_test, y_test):
    """
    Trains a Support Vector Machine (SVM) model on the given training data and evaluates it on the given test data.

    Args:
        x_train (numpy.ndarray): The training data features.
        y_train (numpy.ndarray): The training data labels.
        x_test (numpy.ndarray): The test data features.
        y_test (numpy.ndarray): The test data labels.

    Returns:
        The trained SVM model.
    """
    y_train_mc = np.argmax(y_train, axis=1)
    y_test_mc = np.argmax(y_test, axis=1)
    model = SVC(kernel=os.getenv('SVM_KERNEL', 'rbf'),
                C=float(os.getenv('SVM_C', 1.0)), random_state=42)
    model.fit(x_train, y_train_mc)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test_mc, y_pred)
    print("Accuracy: ", accuracy)

    report = classification_report(y_test_mc, y_pred)
    print(report)

    matrix = confusion_matrix(y_test_mc, y_pred)
    print("Confusion Matrix: \n", matrix)

    pickle.dump(model, open(
        os.getenv('SVM_MODEL_PATH', "./models/svm.pickle.dat"), "wb"))

    print("Saved model to disk")

    return model
