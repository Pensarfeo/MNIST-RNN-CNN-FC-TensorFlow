import tensorflow as tf
import models.layers as layers

keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

def net(data, train=False):
  layers.data_type = data_type
  pool1 = layers.conv(data, 32, 'conv-1')
  pool2 = layers.conv(pool1, 64, 'conv-2')

  pool2_shape = pool2.get_shape().as_list()
  pool_to_fc = [pool2_shape[0], pool2_shape[1] * pool2_shape[2] * pool2_shape[3]]
  reshape = tf.reshape(pool2, pool_to_fc)

  fc1, fc1_weights, fc1_biases = layers.fc(reshape, 512, 'fc1')
  fc1 = tf.nn.relu(fc1)
  if train:
    fc1 = layers.dropout(fc1, keep_prob)
  #tf.summary.histogram("fc1/relu", fc1)
  fc2_logits, fc2_weights, fc2_biases = layers.fc(fc1, 10, 'fc2')

  return [fc2_logits, fc1_weights, fc1_biases, fc2_weights, fc2_biases]