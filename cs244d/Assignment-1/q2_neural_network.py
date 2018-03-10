import numpy as np
import tensorflow as tf

input_data = "/Users/srikanth_m07/Documents/ml_dataset/vision/mnist/"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

x = tf.placeholder(tf.float32, shape=[None, num_input], name="inputs")
y = tf.placeholder(tf.float32, shape=[None, num_classes], name="outputs")

weights = {
    "h1" : tf.Variable(tf.truncated_normal([num_input, n_hidden_1], stddev=1.0)),
    "h2" : tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=1.0)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

def sigmoid(x):
    return 1/(1+tf.exp(-x))

def grad_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def neural_network(x, y):
    #feedforward_propagation
    hiddel_layer_1 = tf.matmul(x, weights['h1'])
    activation_1 = sigmoid(hiddel_layer_1)
    hiddel_layer_2 = tf.matmul(activation_1, weights['h2'])
    activation_2 = sigmoid(hiddel_layer_2)
    output_layer = tf.matmul(activation_2, weights['out'])
    #backward propagation
    error = (y - output_layer) #this is derivative of squared error
    dO = error * grad_sigmoid(error)
    dw2 = weights['h2'] * grad_sigmoid()


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    o = sess.run(backward_propagation(batch_x, batch_y))
    print(o.shape)