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

# plots the image
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary) # shows image
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
               100*np.max(predictions_array),
               class_names[true_label]), color = color)

# makes bar plot
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    # makes bar plot
    thisplot = plt.bar(range(10), predictions_array, color="#777777") # bar graph
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red') # sets color red if incorrect
    thisplot[true_label].set_color('blue')

# import fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# divide by 255 to scale thing to between 0 and 1
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
    plt.xlabel(class_names[train_labels[i]]) # labels X-axis
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
# list of the output layers of all test images

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*3*num_cols, 3*num_rows)) # changes up the figure size
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1) # get the plot ready to plot
  # location of plot, given index and dimensions of the plt
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout() # everything clumped together
plt.show()

img = test_images[1]
# keras models make predictions on a batch at once
# so, we need to add the image to the list
img = (np.expand_dims(img, 0))

predictions_single = model.predict(img)
plot_value_array(1, predictions_single, test_labels)
