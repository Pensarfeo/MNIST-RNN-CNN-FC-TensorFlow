# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.
This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import gzip
import os
import sys
import time
import importlib
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import dataset
import trainer
import utils

import pdb
import scipy.misc

PIXEL_DEPTH = 255
NUM_LABELS = 10

BATCH_SIZE = 50
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 50
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

FLAGS = 1


TRAIN = False
MODEL = 'rnn'
LOGDIR = os.path.join(os.getcwd(), 'tmp', MODEL, 'graph')
SAVE_MODEL_DIR = os.path.join(os.getcwd(), 'saved', MODEL + '.ckpt')
model = importlib.import_module('models.' + MODEL)

def pprint(a, *b, **all):
    print('-----------------------------')
    print(a)
    for x in b:
      print(b)
    print('#############################')


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32

model.data_type = data_type

def main(_):
  train_size = dataset.train_size
  image_size = dataset.image_size
  test_size = dataset.test_size
  #------------------------------------------------------------------
  # Data Training placeholders
  #------------------------------------------------------------------

  train_data_node = tf.placeholder(
    data_type(),
    shape=(BATCH_SIZE, *image_size),
    name='train'
  )

  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

  tf.summary.image('input', train_data_node, BATCH_SIZE)

  #------------------------------------------------------------------
  # Init model
  #------------------------------------------------------------------

  [logits, *weight_and_biases] = model.net(train_data_node, True)
  
  #------------------------------------------------------------------
  # Train, validation & accuracy 
  #------------------------------------------------------------------
  optimizer, predictions, accuracy, prediction, correct_prediction = trainer.main(
    train_labels_node,
    logits,
    weight_and_biases,
    BATCH_SIZE,
    data_type,
    train_size,
    train_labels_node
  )

  #------------------------------------------------------------------
  # Logging
  #------------------------------------------------------------------
  summ = tf.summary.merge_all()
  start_time = time.time()
  saver = tf.train.Saver()

  #------------------------------------------------------------------
  # session
  #------------------------------------------------------------------

  feed_dict_gen = dataset.feed_dict_gen(BATCH_SIZE, train_labels_node, train_data_node)

  if TRAIN:
    # util to clean all prev logs
    writerTrain = utils.fresh_log_writer(LOGDIR, 'train')
    writerValid = utils.fresh_log_writer(LOGDIR, 'validate')

    with tf.Session() as sess:
      # Run all the initializers to prepare the trainable parameters.
      tf.global_variables_initializer().run()

      writerTrain.add_graph(sess.graph)
      writerValid.add_graph(sess.graph)
      print('Initialized!')
      #------------------------------------------------------------------
      # training loop
      #------------------------------------------------------------------
      if TRAIN:
        for step in xrange(int(train_size* NUM_EPOCHS ) // BATCH_SIZE):
          feed_dict = feed_dict_gen(step, 0.5, 'train')[0]
          if step % EVAL_FREQUENCY == 0: # and step!=0:
            start_time = utils.epoch_tracker(step, start_time, BATCH_SIZE / train_size, EVAL_FREQUENCY)
            feed_dict['keep_prob:0'] = 1
            ss = sess.run(summ, feed_dict=feed_dict)
          else:
            __, _, ss, acc = sess.run([optimizer, predictions, summ, accuracy], feed_dict=feed_dict)
          writerTrain.add_summary(ss, step)
        save_path = saver.save(sess, SAVE_MODEL_DIR)

  else:
    #------------------------------------------------------------------
    # testing loop
    #------------------------------------------------------------------
    writerTest =  utils.fresh_log_writer(LOGDIR, 'test')
    with tf.Session() as sess:
      saver.restore(sess, SAVE_MODEL_DIR)
      testPredictions = []
      for step in xrange(int(test_size) // BATCH_SIZE):
        writerTest.add_graph(sess.graph)
        feed_dict, batch_labels, batch_data = feed_dict_gen(step, 1, 'test')
        ss = sess.run(summ, feed_dict=feed_dict)
        # if step % EVAL_FREQUENCY == 0: # and step!=0:
        #   start_time = utils.epoch_tracker(step, start_time, BATCH_SIZE / train_size, EVAL_FREQUENCY)
        accuracyres, cnp, cp, ss = sess.run([accuracy, prediction, correct_prediction, summ], feed_dict=feed_dict)
        misclassified = utils.get_mislabeled_cases(cnp, batch_labels, batch_data, step, BATCH_SIZE)
        writerTest.add_summary(ss, step)
        testPredictions.append(accuracyres)
      pprint(misclassified)
      pprint(numpy.average(testPredictions))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)