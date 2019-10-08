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