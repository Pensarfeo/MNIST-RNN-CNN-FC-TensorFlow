import tensorflow as tf

SEED = 66478  # Set to None for random seed.
data_type = lambda: None


def conv(input, cout, name='conv', const=0.1):
  cin = input.shape.as_list()[-1]
  with tf.name_scope(name):
    weights = tf.Variable(
        tf.truncated_normal(
            [5, 5, cin, cout],  # 5x5 filter, depth 32.
            stddev=0.1,
            seed=SEED,
            dtype=data_type()
        ),
      name='W'
    )
    biases = tf.Variable(tf.constant(const, shape=[cout]), name='B')
    conv = tf.nn.conv2d(
      strides=[1, 1, 1, 1],
      padding='SAME',
      input=input,
      filter=weights,
    )
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
    pool = tf.nn.max_pool(
      relu,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME'
    )
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)
    tf.summary.histogram("activations", relu)
    return pool

import pdb
def fc(input, cout, name='fc'):
  cin = input.shape.as_list()[-1]
  with tf.name_scope(name):
    weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
        [cin, cout],
        stddev=0.1,
        seed=SEED,
        dtype=data_type()
      ),
      name='W'
    )
    biases = tf.Variable(tf.constant(0.1, shape=[cout], dtype=data_type()), name='B')

    output = tf.matmul(input, weights) + biases

    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)
    tf.summary.histogram("activations", output)

    return [output, weights, biases]

def dropout(input, keep_prob):
    return tf.nn.dropout(input, keep_prob, seed=SEED)

def rec(input, n_size, name='rec', const=0.1):
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_size,state_is_tuple=True)
  outputs, states = tf.nn.static_rnn(lstm_cell, input, dtype=data_type())
  tf.summary.histogram("outputs", outputs)
  tf.summary.histogram("states", states)
  return outputs
