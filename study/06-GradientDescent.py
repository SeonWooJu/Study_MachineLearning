import tensorflow as tf

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(-3.0)
# Linear model
hypothesis = X * W
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize: Gradient Descent Magic
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1)


def train_step():
    with tf.GradientTape() as tape:
        global hypothesis
        global cost
        hypothesis = X * W
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    gradients = tape.gradient(cost, [W])
    optimizer.apply_gradients(zip(gradients, [W]))


for step in range(100):
    tf.print(step, W)
    train_step()
