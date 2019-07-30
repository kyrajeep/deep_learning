# Practice from Katacoda.
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
#loaded data in the folder MNIST_data! :)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# these are parameters I can tune :)
image_size = 28
labels_size = 10
learning_rate = 0.01
steps_number = 1000
batch_size = 100

training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

# Variables to be tuned. The weights change constantly so they are represented as variables.
# I dont understand how everything is quite working out yet. But I got the big picture. That's kind of important
W = tf.Variable(tf.truncated_normal([image_size*image_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Build the network (only output layer)
# no activation function yet? :)
output = tf.matmul(training_data, W) + b
#logits = output. you feed the softmax with labels and logits and it spits out the loss! Nice! :)
# Define the loss function
# Did not understand everything about how the softmax works.
# reduce_mean is for taking the average over these sums
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

# Training step to minimize loss. Minimize is a method attached to the Gradient Descent object! 
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

 #Run the training
sess = tf.InteractiveSession()
# this is an important step to initialize all variables.
sess.run(tf.global_variables_initializer())

# the steps number is pre-defined.
for i in range(steps_number):
  # Get the next batch :) returned :) cool! it's automatic
  input_batch, labels_batch = mnist.train.next_batch(batch_size)
  feed_dict = {training_data: input_batch, labels: labels_batch}
    # feed dict is a dictionary to put in. look up again :)

  # Run the training step
  train_step.run(feed_dict=feed_dict)
