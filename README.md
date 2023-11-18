# CNN-Gradient
# Addressing Data Set Imbalance using CNN-Gradient Boosting in Filipino Sign Language Recognition

## Overview

This project focuses on addressing data set imbalance in the Filipino Sign Language Recognition task through a combination of Convolutional Neural Networks (CNNs) and Gradient Boosting. The primary goal is to enhance the model's performance in recognizing signs from the Filipino Sign Language.

## Contents

1. **Model Architecture:**
   - The model architecture is implemented using a combination of Convolutional Neural Networks (CNNs) and Gradient Boosting. The CNN is designed using the ResNet architecture for effective feature extraction from images.

2. **Data Set Imbalance Handling:**
   - To handle the data set imbalance, various techniques are employed, including oversampling, undersampling, or a combination of both. The goal is to ensure that the model is trained on a balanced representation of each class in the Filipino Sign Language.

3. **Evaluation Metrics:**
   - The model is evaluated using standard classification metrics, including accuracy, precision, recall, and the F1 score. These metrics provide insights into the model's overall performance and its ability to correctly identify different sign language classes.

4. **Confusion Matrix and Prediction Visualization:**
   - The confusion matrix is generated to understand the model's performance on individual classes. Additionally, a set of correct and incorrect predictions is visually represented through saved images in the "predictions" folder.

5. **Learning Curve and Accuracy Curve:**
   - Learning curves depict the training and validation loss over epochs, helping to identify potential overfitting or underfitting. Accuracy curves provide insights into the model's accuracy improvement throughout the training process.

6. **Precision-Recall Curve and ROC Curve:**
   - Precision-Recall curves showcase the trade-off between precision and recall, providing a comprehensive evaluation of the model's classification performance. ROC curves visualize the model's true positive rate against false positive rate.

## How to Use

### Environment Setup

Ensure that the required Python environment is set up with the necessary libraries, including TensorFlow, scikit-learn, and matplotlib.

### Model Training

Train the model using the provided CNN architecture and handle data set imbalance through oversampling or undersampling techniques.

### Model Evaluation

Utilize the `h5Evaluators` and  `pickleEvaluators` function to evaluate the model on a test set. This function calculates accuracy, generates classification reports, and saves visualizations, including the confusion matrix and prediction images.

### Graphs and Visualizations

The "graphs" folder contains visualizations such as learning curves, accuracy curves, precision-recall curves, and ROC curves. These graphs offer a detailed analysis of the model's performance.

## Usage

python main.py

```python
