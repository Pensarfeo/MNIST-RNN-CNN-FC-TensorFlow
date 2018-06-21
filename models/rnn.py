import tensorflow as tf
import models.layers as layers

SEED = 66478  # Set to None for random seed.
n_size = 128
keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

def reshapeData(data):
  data_shape = data.get_shape().as_list()
  x = tf.reshape(data, data.get_shape().as_list()[0:3])
  x = tf.transpose(x, [1,0,2])
  x = tf.reshape(x, [-1, data_shape[1]])
  x = tf.split(x, data_shape[1], 0)
  return x

def net(data, train=False, data_type=tf.float16):
  rec = layers.rec(reshapeData(data), n_size, data_type)
  fc1, fc1_weights, fc1_biases = layers.fc(rec[-1], 10, 'fc1', data_type)

  return [fc1, fc1_weights, fc1_biases]