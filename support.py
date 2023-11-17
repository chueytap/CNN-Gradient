import keras
from keras import optimizers
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json


def get_feature_layer(model, data):

    total_layers = len(model.layers)

    fl_index = total_layers-7

    feature_layer_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(index=fl_index).output)

    feature_layer_output = feature_layer_model.predict(data)

    return feature_layer_output


def load_resnet():

    json_file = open("./models/resnet.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./models/resnet.h5")

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        loss='categorical_crossentropy',
        metrics=[
            metrics.CategoricalAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
        ]

    )

    return model


class F1ScoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        precision = logs.get('precision')
        recall = logs.get('recall')
        if precision and recall:
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"\nF1 Score at end of epoch {epoch}: {f1_score}")
