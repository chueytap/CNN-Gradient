from dataset import load_dataset
import os
from h5Evaluators import h5Evaluators

from model import gradient_model, resnet_model, smote_model, svm_model
from support import get_feature_layer, load_resnet
from pickleEvaluators import gradient_evaluator, svm_evaluator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":

    dataset_path = os.getenv('DATASET_PATH', './dataset_custom')
    if dataset_path is None:
        print("Error: DATASET_PATH environment variable not set.")
        exit(1)
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_path)

    print("Choose an option:")
    print("1. Train and Save Models")
    print("2. Load Models and Evaluate Metrics")

    option = input("Enter your choice (1 or 2): ")

    

    if option == '1':
        print("Features extracted from training data")
        print("Training and saving H5-based ResNet-50 models...")
        # print("Training ResNet-50 model...")
        resnet_model(x_train, y_train, x_test, y_test)
        # print("Training Smote with ResNet-50 model...")
        smote_model(x_train, y_train, x_test, y_test)
        # print("Training and saving H5-based ResNet-50 models complete\n")
        # cnn_model = load_resnet()
        # print("Loaded CNN model from disk")
        # x_test_cnn = get_feature_layer(cnn_model, x_test)
        # print("Features extracted from test data\n")
        # x_train_cnn = get_feature_layer(cnn_model, x_train)
        # Train and save models
        # print("Training and saving Pickle-based ResNet-50 models...")
        # print("Training and saving Gradient Boosting model...")
        # gradient_model(x_train_cnn, y_train, x_test_cnn, y_test)
        # print("Built and saved Gradient Boosting Model")
        # print("Training and saving SVM model...")
        # svm_model(x_train_cnn, y_train, x_test_cnn, y_test)
        # print("Built and saved SVM Model")

    elif option == '2':
        cnn_model = load_resnet()
        print("Loaded CNN model from disk")

        x_test_cnn = get_feature_layer(cnn_model, x_test)
        print("Features extracted from test data\n")

        x_train_cnn = get_feature_layer(cnn_model, x_train)
        # Load and evaluate metrics of the ResNet-50 model
        print("Evaluating H5-based ResNet-50 model...")
        print("Evaluating ResNet-50 model...")
        model_path = "./models/resnet.h5"
        h5Evaluators(model_path, x_test, y_test, 'resnet')
        print("Evaluating Smote with ResNet-50 model...")
        model_path = "./models/smote.h5"
        h5Evaluators(model_path, x_test, y_test, 'smote')

        print("Evaluating Pickle-based ResNet-50 model...")
        print("Evaluating Gradient Boosting model...")
        gradient_evaluator(x_test, y_test, x_test_cnn)
        print("Evaluating SVM model...")
        svm_evaluator(x_test, y_test, x_test_cnn)
    else:
        print("Invalid option. Please choose 1 or 2.")
