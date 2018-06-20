import tensorflow as tf
import models.layers as layers

keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

def net(data, train=False):
  layers.data_type = data_type
  pool1 = layers.conv(data, 32, 'conv-1')
  pool2 = layers.conv(pool1, 64, 'conv-2')

  data_shape = data.get_shape().as_list()
  data_to_fc = [data_shape[0], data_shape[1] * data_shape[2] * data_shape[3]]
  reshape = tf.reshape(data, data_to_fc)
  
  fc1, fc1_weights, fc1_biases = layers.fc(reshape, 512, 'fc1')
  # fc1 = tf.nn.relu(fc1)
  fc2, fc2_weights, fc2_biases = layers.fc(fc1, 512//2, 'fc2')
  # fc2 = tf.nn.relu(fc2)
  fc3, fc3_weights, fc3_biases = layers.fc(fc2, 512//4, 'fc3')
  # fc3 = tf.nn.relu(fc3)
  fc4, fc4_weights, fc4_biases = layers.fc(fc3, 10, 'fc4')

  # if train:
  #   fc1 = layers.dropout(fc1, keep_prob)
  #tf.summary.histogram("fc1/relu", fc1)

  return [fc4, fc1_weights, fc1_biases, fc2_weights, fc2_biases, fc3_weights, fc3_biases, fc4_weights, fc4_biases]