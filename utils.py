
import time
import shutil
from six.moves import xrange
import os
import tensorflow as tf
import time

# show how much time has ellapsed
def epoch_tracker(step, prevTime, bs_ts, EVAL_FREQUENCY):
  # bs_ts = BATCH_SIZE / train_size
  elapsed_time = time.time() - prevTime
  prevTime = time.time()
  print('Step %d (epoch %.2f), %.1f ms' %(step, float(step) * bs_ts, 1000 * elapsed_time / EVAL_FREQUENCY))
  return prevTime

# how misclassified cases and output image
def get_mislabeled_cases(predictions, labels, data, step, BATCH_SIZE):
  misclassifiedClasses = []
  for i in xrange(0, predictions.size):
    if (predictions[i]!=labels[i]):
      newMiss = [predictions[i], labels[i], (step * BATCH_SIZE)+i]
      misclassifiedClasses.append(newMiss)
      # image_name = './'+str(predictions[i])+'_'+str(labels[i])+'_'+str((step * BATCH_SIZE)+i)+'.jpg'
      # scipy.misc.toimage( list(map(lambda x: x.flatten(), data[i]))).save(image_name)
      print(newMiss)
  return misclassifiedClasses

def fresh_log_writer(*paths):
  writer_dir = os.path.join(*paths)
  # if os.path.isdir(writer_dir):
  #     shutil.rmtree(writer_dir)
  # print('writing new logs to')
  # print(writer_dir)
  # print('------------------------------------------')
  # time.sleep(1)
  return tf.summary.FileWriter(writer_dir)