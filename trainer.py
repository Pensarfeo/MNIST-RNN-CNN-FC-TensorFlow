import tensorflow as tf
import pdb

def main(labels, logits, weight_and_biases, BATCH_SIZE, data_type, train_size, train_labels_node):
  """
    set up the loss and optimization part of the graph
  """
  # L2 loss is used
  # I am not sure if sparse_softmax_cross_entropy_with_logits referes to:
  # https://arxiv.org/abs/1602.02068

  with tf.name_scope('cost'):
    cost = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
      ),
      name='cost'
    )
    
    tf.summary.scalar("cost", cost)
    losses_L2 = map(lambda x: tf.nn.l2_loss(x) , weight_and_biases)
    regularizers = tf.reduce_sum(list(losses_L2))
    loss = cost + 5e-4 * regularizers

  batch = tf.Variable(0, dtype=data_type())
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)

  with tf.name_scope('train'):
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
  predictions = tf.nn.softmax(logits)

  # we only use tf operators cause we want to get them as the output of sess.run
  with tf.name_scope('accuracy'):
    prediction = tf.argmax(predictions, 1)
    correct_prediction = tf.equal(prediction, train_labels_node)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, data_type()))
    accuracySummary = tf.summary.scalar('accuracy', accuracy)

  return [optimizer, predictions, accuracy, prediction, correct_prediction]