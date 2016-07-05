import tensorflow as tf
import numpy as np

#plotting libs

from matplotlib import pyplot
from PIL import Image

#load data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

print("The training set is:")
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print("The test set is: ")
print(mnist.test.images.shape)
print(mnist.test.labels.shape)
print("Validation set")
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))