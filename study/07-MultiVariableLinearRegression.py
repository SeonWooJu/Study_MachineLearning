import tensorflow as tf
import numpy as np

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

# Define placeholders for input data and labels
x1 = tf.constant(x1_data, dtype=tf.float32)
x2 = tf.constant(x2_data, dtype=tf.float32)
x3 = tf.constant(x3_data, dtype=tf.float32)
Y = tf.constant(y_data, dtype=tf.float32)

# Define variables for weights and bias
W1 = tf.Variable(tf.random.normal([1]), name='weight1', dtype=tf.float32)
W2 = tf.Variable(tf.random.normal([1]), name='weight2', dtype=tf.float32)
W3 = tf.Variable(tf.random.normal([1]), name='weight3', dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), name='bias', dtype=tf.float32)

# Our hypothesis for the linear model X1 * W1 + X2 * W2 + X3 * W3 + b
hypothesis = x1 * W1 + x2 * W2 + x3 * W3 + b

# Define the cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Create a GradientDescentOptimizer
optimizer = tf.optimizers.SGD(learning_rate=1e-5)

# Minimize the cost
for step in range(2001):
    with tf.GradientTape() as tape:
        hypothesis = x1 * W1 + x2 * W2 + x3 * W3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    gradients = tape.gradient(cost, [W1, W2, W3, b])
    optimizer.apply_gradients(zip(gradients, [W1, W2, W3, b]))

    if step % 10 == 0:
        tf.print(step, "Cost: ", cost, "\nInPrediction: \n", hypothesis)
