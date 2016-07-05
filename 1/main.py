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

# Helps to understand how accurate something is
def accuracy(predictions,labels):
    return (100.0 * np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0]);


def multinomialLogisticRegression():

    #PART 1: Describe the graph

    graph = tf.Graph()
    with graph.as_default():

        #Graph variables

        batch_size = 128 #small batch that will be chosen randomly
        beta = 0.001 #regularization
        image_size = 28
        num_labels = 10

        learning_rate = 0.5 # for Gradient Descent

        #Create a placeholder that will be fed with training minibatches at run time

        tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,num_labels))
        tf_valid_dataset = tf.constant(mnist.validation.images)
        tf_test_dataset = tf.constant(mnist.test.images)

        #Weights and biases from output/logit layer
        w_logit = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
        b_logit = tf.Variable(tf.zeros([num_labels]))


        #logit = predictions, we make prediction and calculate the error

        logits = model(tf_train_dataset, w_logit, b_logit) # make a prediction

        # having made the prediction we need to calculate how correct it is
        # the labels are the correct answers and we compare them to the prediction that were made
        # softmax_cross_entropy is a common error function

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,tf_train_labels))
        regularized_loss = tf.nn.l2_loss(w_logit)
        total_loss = loss + beta * regularized_loss

        # Now when we know the error we can use Gradient Descent to adjust

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(total_loss);

        # Predictions for the training, validation and test data

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, w_logit, b_logit))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, w_logit, b_logit))

        # PART 2: Train

        num_steps = 5000

        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            for step in range(num_steps):

                # Generate a minibatch
                batch_data, batch_labels = mnist.train.next_batch(batch_size);

                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

                #RUN IN THE GRAPH
                _, l, predictions = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)

                if(step % 500 == 0):
                    print("Minibatch loss at step %d: %f" % (step,l))
                    print("Minibatch accuracy: %f" % accuracy(predictions,batch_labels))
                    print("Validation accuracy: %f" % accuracy(valid_prediction.eval(),mnist.validation.labels))
                    print("Test accuracy: %f" % accuracy(test_prediction.eval(),mnist.test.labels))
                    print("------\n\n")



multinomialLogisticRegression()

