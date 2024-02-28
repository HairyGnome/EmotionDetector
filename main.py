import datetime
import os

import numpy as np

import tensorflow as tf
from data_processor import DataProcessor

img_rows, img_cols = 48, 48
input_shape = (img_rows, img_cols, 1)

train_data_path = os.path.join(os.curdir, "data\\train")
valid_data_path = os.path.join(os.curdir, "data\\valid")
test_data_path = os.path.join(os.curdir, "data\\test")

data_processor = DataProcessor()
x_train, y_train = data_processor.read_images(train_data_path)
x_valid, y_valid = data_processor.read_images(valid_data_path)
x_test, y_test = data_processor.read_images(test_data_path)


x_train = np.expand_dims(x_train, axis=-1)
x_valid = np.expand_dims(x_valid, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    zoom_range=0.2
)

datagen.fit(x_train)
datagen.fit(x_valid)

num_classes = 6

batch_size = 64
epochs = 100


physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU device found:", physical_devices[0])
else:
    print("No GPU devices found, using CPU.")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv2d_last'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(datagen.flow(x_train, y_train,
          batch_size=batch_size),
          epochs=epochs,
          validation_data=datagen.flow(x_valid, y_valid, batch_size=batch_size),
          callbacks=[tensorboard_callback])

print("Evaluation")
test_loss, test_accuracy = model.evaluate(x_test, y_test)

if input("Save model? [Y/N] ").capitalize() == 'Y':
    model.save(f'saved_models/%.4f' % test_accuracy, overwrite=True)

