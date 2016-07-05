import tensorflow as tf
import numpy as np

#plotting libs

from matplotlib import pyplot as plt
from PIL import Image

#load data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

def printSetInfo():
    print("The training set is:")
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)
    print("The test set is: ")
    print(mnist.test.images.shape)
    print(mnist.test.labels.shape)
    print("Validation set")
    print(mnist.validation.images.shape)
    print(mnist.validation.labels.shape)


def runTFTest():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

    a = tf.constant(10)
    b = tf.constant(32)
    print(sess.run(a + b))

def plotImage(index):
    image2d = np.reshape(mnist.train.images[index],[28,28])
    plt.imshow(image2d,cmap="Greys_r")
    plt.colorbar()
    plt.title('plot for sin(x)+sin(y)')
    plt.show()



def model(data,w_logit,b_logit):
    #Assemble the NN
    return tf.matmul(data,w_logit)+b_logit


def multinomialLogisticRegression():
    #PART 1: Describe the graph
    graph = tf.Graph()
    with graph.as_default():

        #Graph variables
        batch_size = 128
        beta = 0.001 #regularization
        image_size = 28
        num_labels = 10

        #Create a placeholder that will be fed with training minibatches at run time
        tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,num_labels))
        tf_valid_dataset = tf.constant(mnist.validation.images)
        tf_test_dataset = tf.constant(mnist.test.images)

        #Weights and biases from output/logit layer
        w_logit = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
        b_logit = tf.Variable(tf.zeros([num_labels]))



plotImage(5)

