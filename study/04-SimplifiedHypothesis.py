import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

# Define the Weight (W) as a TensorFlow Variable
W = tf.Variable(tf.random.normal([1]), name="weight", dtype=tf.float32)


# Our hypothesis for the linear model X * W
def hypothesisFunc(x):
    return x * W


# Define the cost/loss function
def cost_function(hypothesis, y):
    return tf.reduce_mean(tf.square(hypothesis - y))


# Variables for plotting cost function
W_val = []
cost_val = []

# Range of values for W
for i in range(-30, 50):
    feed_W = i * 0.1
    W.assign([feed_W])  # Assign the value to W
    curr_cost = cost_function(hypothesisFunc(X), Y).numpy()
    W_val.append(feed_W)
    cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val, cost_val)
plt.xlabel('W')
plt.ylabel('Cost')
plt.show()
