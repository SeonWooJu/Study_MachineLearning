import tensorflow as tf

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, TensorFlow!");

# Versions above 2.0.0 do not use "tensorflow.Session()"

# run the op end get result
tf.print(hello)
