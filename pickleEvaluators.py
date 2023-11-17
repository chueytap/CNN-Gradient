

import matplotlib.pyplot as plt

import os
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import matplotlib.pyplot as plt


# Save correctly predicted images to a folder
class_mapping = {
    "afternoon": 0,
    "are": 1,
    "evening": 2,
    "good": 3,
    "hard": 4,
    "hello": 5,
    "how": 6,
    "morning": 7,
    "of-hearing": 8,
    "thank": 9,
    "today": 10,
    "tomorrow": 11,
    "understand": 12,
    "you": 13
}


def gradient_evaluator(x_test, y_test, x_test_cnn):
    """
    Evaluates the performance of a gradient model on a given test set.

    Args:
    x_test (numpy.ndarray): The test set features.
    y_test (numpy.ndarray): The test set labels.
    x_test_cnn (numpy.ndarray): The test set features reshaped for CNN.

    Returns:
    None
    """
    loaded_catboost_model = pickle.load(
        open("./models/gradient.pickle.dat", "rb"))
    print("Loaded Gradient model from disk")

    y_pred_gradient = loaded_catboost_model.predict(x_test_cnn)

    y_test_mc = np.argmax(y_test, axis=1)

    accuracy_catboost = accuracy_score(y_test_mc, y_pred_gradient)
    print("Gradient Model Accuracy: ", accuracy_catboost)

    report_catboost = classification_report(y_test_mc, y_pred_gradient)
    print(report_catboost)

    matrix_catboost = confusion_matrix(y_test_mc, y_pred_gradient)
    print("Gradient Model Confusion Matrix: \n", matrix_catboost)

    correct_predictions_folder = "./predictions_gradient/correct_prediction"
    wrong_predictions_folder = "./predictions_gradient/wrong_prediction"
    if not os.path.exists(correct_predictions_folder):
        os.makedirs(correct_predictions_folder)
    if not os.path.exists(wrong_predictions_folder):
        os.makedirs(wrong_predictions_folder)

    image_distributor(x_test, y_test_mc, y_pred_gradient,
                      correct_predictions_folder, wrong_predictions_folder)


def svm_evaluator(x_test, y_test, x_test_cnn):
    """
    Evaluates the performance of the SVM model on the test set.

    Args:
    x_test (numpy.ndarray): The test set features.
    y_test (numpy.ndarray): The test set labels.
    x_test_cnn (numpy.ndarray): The test set features in CNN format.

    Returns:
    None
    """
    # Load the SVM model from the saved file
    loaded_svm_model = pickle.load(
        open("./models/svm.pickle.dat", "rb"))
    print("Loaded SVM model from disk")

    # Make predictions on the test set using the loaded SVM model
    y_pred_svm = loaded_svm_model.predict(x_test_cnn)

    # Convert one-hot encoded labels to integers for evaluation
    y_test_mc = np.argmax(y_test, axis=1)

    # Calculate and print the accuracy
    accuracy_svm = accuracy_score(y_test_mc, y_pred_svm)
    print("SVM Model Accuracy: ", accuracy_svm)

    # Print precision, recall, F1 score, etc.
    report_svm = classification_report(y_test_mc, y_pred_svm)
    print(report_svm)

    # Print the confusion matrix
    matrix_svm = confusion_matrix(y_test_mc, y_pred_svm)
    print("SVM Model Confusion Matrix: \n", matrix_svm)

    correct_predictions_folder = "./predictions_svm/correct_prediction"
    wrong_predictions_folder = "./predictions_svm/wrong_prediction"
    if not os.path.exists(correct_predictions_folder):
        os.makedirs(correct_predictions_folder)
    if not os.path.exists(wrong_predictions_folder):
        os.makedirs(wrong_predictions_folder)

    image_distributor(x_test, y_test_mc, y_pred_svm,
                      correct_predictions_folder, wrong_predictions_folder)


def image_distributor(x_test, y_test_mc, y_pred_svm, correct_predictions_folder, wrong_predictions_folder):
    for i in range(len(y_test_mc)):
        if y_test_mc[i] == y_pred_svm[i]:
            predicted_label = list(class_mapping.keys())[list(
                class_mapping.values()).index(y_pred_svm[i])]
            if not os.path.exists(os.path.join(correct_predictions_folder, predicted_label)):
                os.makedirs(os.path.join(
                    correct_predictions_folder, predicted_label))
            plt.imsave(os.path.join(correct_predictions_folder, predicted_label,
                                    f"image_{i}.png"), x_test[i])
        else:
            predicted_label = list(class_mapping.keys())[list(
                class_mapping.values()).index(y_pred_svm[i])]
            if not os.path.exists(os.path.join(wrong_predictions_folder, predicted_label)):
                os.makedirs(os.path.join(
                    wrong_predictions_folder, predicted_label))
            true_label = list(class_mapping.keys())[list(
                class_mapping.values()).index(y_test_mc[i])]
            plt.imsave(os.path.join(wrong_predictions_folder, predicted_label,
                                    f"true_label_{true_label}_predicted_label_{predicted_label}_image_{i}.png"), x_test[i])
