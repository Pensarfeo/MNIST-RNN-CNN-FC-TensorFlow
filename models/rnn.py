import tensorflow as tf
import models.layers as layers

SEED = 66478  # Set to None for random seed.
data_type = lambda: None
n_size = 128
keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
def reshapeData(data):
  data_shape = data.get_shape().as_list()
  x = tf.reshape(data, data.get_shape().as_list()[0:3])
  x = tf.transpose(x, [1,0,2])
  x = tf.reshape(x, [-1, data_shape[1]])
  x = tf.split(x, data_shape[1], 0)
  return x

def net(data, train=False):
  layers.data_type = data_type
  rec = layers.rec(reshapeData(data), n_size)
  import pdb
  pdb.set_trace()
  fc1, fc1_weights, fc1_biases = layers.fc(rec[-1], 10, 'fc1')
  
  return [fc1, fc1_weights, fc1_biases]