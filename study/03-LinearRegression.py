import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)
# TensorFlow 2.0.0 and later versions do not use the function "optimizer.minimize()"


# Gradient function
def train_step():
    with tf.GradientTape() as tape:
        global hypothesis
        global cost
        hypothesis = x_train * W + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    gradients = tape.gradient(cost, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# Fit the Line
for step in range(2001):
    train_step()
    if step % 20 == 0:
        tf.print(step, cost, W, b)
