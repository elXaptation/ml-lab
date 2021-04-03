#!/usr/bin/python3
import tensorflow as tf
import json
import inspect

mnist = tf.keras.datasets.mnist
input("Checkpoint: 01 - Define dataset complete")
(x_train, y_train),(x_test, y_test) = mnist.load_data()
input("Checkpoint: 02 - dataset load complete ")
x_train, x_test = x_train / 255.0, x_test / 255.0
input("Checkpoint: 03 - train and test data modified complete")

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
input("Checkpoint: 04 - Model definition complete")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
input("Checkpoint: 05 - Model compile complete")

model.fit(x_train, y_train, epochs=5)
i = 1
input("Checkpoint: 06 - Model training complete")
model.evaluate(x_test, y_test)
