import tensorflow as tf
import numpy as np

x_data = np.array([
    [73., 80., 75.],
    [93., 88., 93.],
    [89., 91., 90.],
    [96., 98., 100.],
    [73., 66., 70.],
    [53., 46., 55.]
], dtype=np.float32)
y_data = np.array([
    [152.],
    [185.],
    [180.],
    [196.],
    [142.],
    [101.]
], dtype=np.float32)

# Define the Weight (W) and bias (b) as TensorFlow Variables
W = tf.Variable([[0], [0], [0]], name='weight', dtype=tf.float32)
b = tf.Variable([1], name='bias', dtype=tf.float32)

print("Final W:", W.numpy())
print("Final b:", b.numpy())

# Hypothesis
def hypothesis(x):
    return tf.matmul(x, W) + b


# Define the cost/loss function
def cost_function(hypothesis, y):
    return tf.reduce_mean(tf.square(hypothesis - y))


# Optimizer (Gradient Descent)
learning_rate = 1e-5
optimizer = tf.optimizers.SGD(learning_rate)

step = 0
# Training loop
while 1:
    step += 1
    with tf.GradientTape() as tape:
        cost = cost_function(hypothesis(x_data), y_data)
    gradients = tape.gradient(cost, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if step % 10 == 0:
        tf.print(step, "Cost:", cost, "Prediction:\n", hypothesis(x_data))

    if cost <= 0:
        break

# Print the final W and b values
print("Final W:", W.numpy())
print("Final b:", b.numpy())

