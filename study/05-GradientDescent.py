import tensorflow as tf
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(1.64462376, name='weight')
X = tf.Variable(tf.random.normal([1, 3]), dtype=tf.float32)
Y = tf.Variable(tf.random.normal([1, 3]), dtype=tf.float32)


# Our hypothesis for linear model X * W
def hypothesisFunc():
    return X * W


# cost/loss function
def costFunc(hypothesis):
    return tf.reduce_sum(tf.square(hypothesis - Y))


# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
def minimizeFunc():
    learning_rate = 0.1
    gradient = tf.reduce_mean((W * X - Y) * X)
    descent = W - learning_rate * gradient
    return W.assign(descent)


# Initializes global variables in the graph
for step in range(21):
    X.assign([x_data])
    Y.assign([y_data])
    update = minimizeFunc()
    tf.print(step, costFunc(hypothesisFunc()), W)
