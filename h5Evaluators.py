
import tensorflow as tf
from tensorflow.keras import optimizers, metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from model import build_resnet_model

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


def h5Evaluators(model_path, x_test, y_test, name):

    # Load the model from the h5 file
    model = build_resnet_model()
    model.load_weights(model_path)

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        loss='categorical_crossentropy',
        metrics=[
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
        ]
    )
    print("Evaluating the model on the test set...")
    # Resize the test set to the input shape expected by the model
    x_test_resized = tf.image.resize(x_test, (128, 128))
    y_pred = model.predict(x_test_resized)

    # Convert one-hot encoded labels to integers for evaluation
    y_test_mc = np.argmax(y_test, axis=1)
    y_pred_mc = np.argmax(y_pred, axis=1)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test_mc, y_pred_mc)
    print("Accuracy: ", accuracy)

    # Print precision, recall, F1 score, etc.
    report = classification_report(y_test_mc, y_pred_mc)
    print(report)

    # Print the confusion matrix
    matrix = confusion_matrix(y_test_mc, y_pred_mc)
    print("Confusion Matrix: \n", matrix)

    # Define the path to the "predictions" folder
    predictions_path = "./predictions_" + name

    # Define the names of the subfolders inside "predictions"
    correct_prediction_path = os.path.join(
        predictions_path, "correct_prediction")
    wrong_prediction_path = os.path.join(predictions_path, "wrong_prediction")

    # Create the "predictions" folder and the subfolders inside it
    os.makedirs(correct_prediction_path, exist_ok=True)
    os.makedirs(wrong_prediction_path, exist_ok=True)

    # Reverse the class mapping to get a mapping from integer labels to class names
    reverse_class_mapping = {v: k for k, v in class_mapping.items()}

    # Initialize a counter for the number of incorrect predictions
    incorrect_count = 0

    # Loop through the test set and save each image to the corresponding folder
    for i in range(len(x_test)):
        image = x_test[i]
        true_label = y_test_mc[i]
        predicted_label = y_pred_mc[i]

        # Define the image filename
        image_filename = f"image_{i}.png"

        if true_label == predicted_label:
            # Save the image to the "correct_prediction" folder
            image_path = os.path.join(
                correct_prediction_path, reverse_class_mapping[true_label], image_filename)
        else:
            # Increment the counter for the number of incorrect predictions
            incorrect_count += 1
            # Save the image to the "wrong_prediction" folder
            image_path = os.path.join(
                wrong_prediction_path, reverse_class_mapping[true_label], image_filename)

        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Save the image
        plt.imsave(image_path, image)

    print("Number of incorrect predictions:", incorrect_count)
