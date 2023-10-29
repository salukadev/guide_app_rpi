import tensorflow.compat.v1  as tf
tf.disable_v2_behavior()
# Hand-made leaky relu
def leaky_relu(x, alpha=0.2):
  return tf.maximum(x, alpha * x)

# 2D convolution wrapper
def conv2d_leaky(x, kernel_shape, bias_shape, strides=1, relu=True, padding='SAME'):
  # Conv2D
  weights = tf.get_variable("weights", kernel_shape, initializer=tf.initializers.glorot_uniform(), dtype=tf.float32)
  biases = tf.get_variable("biases", bias_shape, initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
  output = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding=padding)
  output = tf.nn.bias_add(output, biases)
  # ReLU (if required)
  if relu:
    output = leaky_relu(output, 0.2)
  return output

# 2D deconvolution wrapper
def deconv2d_leaky(x, kernel_shape, bias_shape, strides=1, relu=True, padding='SAME'):
  # Conv2D
  weights = tf.get_variable("weights", kernel_shape, initializer=tf.initializers.glorot_uniform(), dtype=tf.float32)
  biases = tf.get_variable("biases", bias_shape, initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
  x_shape = tf.shape(x)
  outputShape = [x_shape[0],x_shape[1]*strides,x_shape[2]*strides,kernel_shape[2]]  
  output = tf.nn.conv2d_transpose(x, weights, output_shape=outputShape, strides=[1, strides, strides, 1], padding=padding)
  output = tf.nn.bias_add(output, biases)
  # ReLU (if required)
  if relu:
    output = leaky_relu(output, 0.2)
  return output    
