# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:50:36 2019

@author: henry
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# import fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255
test_images = test_images / 255

# display images
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# make model with 3 layers
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')])

# compiling the model
# accuracy: model performance evaluated based on number of images correctly classified
# optimizer: how model is updated based on data and its loss function
# loss function: what function the model wants to minimize
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# start training
model.fit(train_images, train_labels, epochs=10)

# evaluate test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest Accuracy:', test_acc)

# see the predictions
predictions = model.predict(test_images)
