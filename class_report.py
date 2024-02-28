import datetime
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import binarize

from data_processor import DataProcessor

class_names = ["angry", "neutral", "fearful", "happy", "sad", "surprised"]
path = "data\\test"
model_path = "saved_models\\0.5853"


def gather_images():
    data_processor = DataProcessor()
    x_test, y_test = data_processor.read_images(path)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def load_model():
    return tf.keras.models.load_model(f'{model_path}')


def make_predictions(x_test, model):
    y_pred = []
    for img in x_test:
        prediction = model.predict((1, img))
        y_pred.append(prediction)

    return np.array(y_pred)


def create_classification_report(y_test, y_pred):
    return classification_report(y_test, y_pred)


test_x, test_y = gather_images()
test_y = np.array(test_y)
model = load_model()

pred_y = []
for i in range(len(test_x)):
    pred_y.append(model.predict(np.expand_dims(test_x[i], axis=0)))

pred_y = np.array(pred_y)


threshold = 0.5

n_samples, n_classes = test_y.shape
test_y_2d = test_y.reshape(n_samples, n_classes)
pred_y_2d = pred_y.reshape(n_samples, n_classes)


binary_test_y = binarize(test_y_2d, threshold=threshold)
binary_pred_y = binarize(pred_y_2d, threshold=threshold)


binary_test_y = binary_test_y.astype(int)
binary_pred_y = binary_pred_y.astype(int)



report = create_classification_report(binary_test_y, binary_pred_y)
print(report)
