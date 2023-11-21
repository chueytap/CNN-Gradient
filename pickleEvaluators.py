

import json
import matplotlib.pyplot as plt

import os
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

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
    name="gradient"

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

    confusion_matrix_path = os.path.join(
    "./website/assets/graphs/", name, f"{name}_confusion_matrix.png")

      # Save confusion matrix as an image
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_catboost, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=list(class_mapping.keys()), yticklabels=list(class_mapping.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(
        "./website/assets/graphs/", name, f"{name}_confusion_matrix.png")
    os.makedirs(os.path.dirname(confusion_matrix_path), exist_ok=True)
    plt.savefig(confusion_matrix_path)


    # Save precision, recall, and f1 score graphs for each class
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test_mc, y_pred_gradient, average=None, labels=np.unique(y_test_mc)
    )
    # Plot bar graphs for accuracy, precision, recall per class
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [precision, recall, f1_score]

    # Calculate averages for precision, recall, and f1-score
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_score_avg = np.mean(f1_score)

    # Plot bar graphs for accuracy, precision, recall per class
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [precision, recall, f1_score]

    metrics_dict = {
        'name': name,
        'accuracy': accuracy_catboost,
        # Convert NumPy array to list for JSON serialization
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1_score.tolist(),
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_score_avg': f1_score_avg
    }

    # Convert metrics dictionary to JSON
    metrics_json = json.dumps(metrics_dict, indent=4)

    # Save metrics dictionary as a JSON file
    metrics_json_path = os.path.join(
        "./website/assets/metrics/",f"{name}_metrics.json")
    os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)
    with open(metrics_json_path, 'w') as json_file:
        json.dump(metrics_dict, json_file, indent=4)

    for metric_name, metric_values in zip(metrics_names, metrics_values):
        plt.figure(figsize=(12, 6))
        bars = plt.bar(list(class_mapping.keys()), metric_values, color=['maroon','orange','lightpink'])
        plt.xlabel('Classes')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} per Class')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
         # Adding labels to the bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.02, f'{value:.2f}', ha='center', va='bottom')

        metric_path = os.path.join(
            "./website/assets/graphs/", name, f"{name}_{metric_name.lower()}")
        os.makedirs(os.path.dirname(metric_path), exist_ok=True)
        plt.savefig(metric_path)

    correct_predictions_folder = "./predictions_gradient/correct_prediction"
    wrong_predictions_folder = "./predictions_gradient/wrong_prediction"
    if not os.path.exists(correct_predictions_folder):
        os.makedirs(correct_predictions_folder)
    if not os.path.exists(wrong_predictions_folder):
        os.makedirs(wrong_predictions_folder)

    image_distributor(x_test, y_test_mc, y_pred_gradient,
                      correct_predictions_folder, wrong_predictions_folder)


def svm_evaluator(x_test, y_test, x_test_cnn):
    name="svm"

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

    # Save confusion matrix
    confusion_matrix_path = os.path.join(
        "./website/assets/graphs/", name, f"{name}_confusion_matrix.png")
      # Save confusion matrix as an image
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_svm, annot=True, fmt="d", cmap="BuPu",
                xticklabels=list(class_mapping.keys()), yticklabels=list(class_mapping.keys()))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(
        "./website/assets/graphs/", name, f"{name}_confusion_matrix.png")
    os.makedirs(os.path.dirname(confusion_matrix_path), exist_ok=True)
    plt.savefig(confusion_matrix_path)


      # Calculate precision, recall, and f1-score per class
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test_mc, y_pred_svm, average=None, labels=list(class_mapping.values())
    )

     # Plot bar graphs for accuracy, precision, recall per class
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [precision, recall, f1_score]

     # Calculate averages for precision, recall, and f1-score
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    f1_score_avg = np.mean(f1_score)

    metrics_dict = {
        'name': name,
        'accuracy': accuracy_svm,
        # Convert NumPy array to list for JSON serialization
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1_score.tolist(),
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_score_avg': f1_score_avg
    }

    # Convert metrics dictionary to JSON
    metrics_json = json.dumps(metrics_dict, indent=4)

    # Save metrics dictionary as a JSON file
    metrics_json_path = os.path.join(
        "./website/assets/metrics/",f"{name}_metrics.json")
    os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)
    with open(metrics_json_path, 'w') as json_file:
        json.dump(metrics_dict, json_file, indent=4)

    for metric_name, metric_values in zip(metrics_names, metrics_values):
        plt.figure(figsize=(12, 6))
        bars =  plt.bar(list(class_mapping.keys()), metric_values, color=['orange','lightgreen','lightpink'])
        plt.xlabel('Classes')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} per Class')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        # Adding labels to the bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.02, f'{value:.2f}', ha='center', va='bottom')

        metric_path = os.path.join(
            "./website/assets/graphs/", name, f"{name}_{metric_name.lower()}")
        os.makedirs(os.path.dirname(metric_path), exist_ok=True)
        plt.savefig(metric_path)

    
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
