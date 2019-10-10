# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:01:12 2019

@author: henry
"""

# program takes in text, classifies it as positive or negative movie reivew
# binary classification

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# splits training data into 60% training and 40% validation, the rest of
# data becomes test data
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True) 

# train examples batch (example of a single batch)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# 1. how to represent the text
# 2. number of layers to use in model
# 3. how many hidden units for each layer?

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3]) -> example of layer output on one training example

model = tf.keras.Sequential()
model.add(hub_layer) # hub_layer, uses pre-trained saved model to map sentence onto 
# embedding vector
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary() # prints stats for the model

# assign error function, optimizer, and method of evaluation to model
# binary_crossentropy
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train data, 20 epochs, 20 iterations over all training data
# monitor loss on validation set
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)




