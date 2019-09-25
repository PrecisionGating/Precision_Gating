from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np

def get_trainable_vars():
  '''Gets trainable variables and also adds batch norm moving avgs'''
  def is_moving_average_var(v):
    return 'batch_normalization' in v.name and '/moving_' in v.name

  g = tf.get_default_graph()
  tvars = g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  vars = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  return [v for v in vars if (v in tvars or is_moving_average_var(v))]

def save_vars(filename, keys=tf.GraphKeys.TRAINABLE_VARIABLES):
  '''Save vars to numpy archive'''
  tvars = get_trainable_vars()
  _tvars = [v.eval() for v in tvars]
  logging.info('Saving %d weights to %s' % (len(_tvars), '_model.npy'))
  for v in tvars:
    print(' ', v.name)
  np.save(filename, _tvars)

def load_vars(filename, keys=tf.GraphKeys.TRAINABLE_VARIABLES):
  '''Load vars from numpy archive'''
  logging.info("Loading weights from %s" % (filename))
  _tvars = np.load(filename)

  tvars = get_trainable_vars()
  for v in tvars:
    print(' ', v.name)

  assert(len(tvars) == len(_tvars))
  for v, _v in zip(tvars, _tvars):
    assert(v.get_shape() == _v.shape)
    tf.assign(v, _v).eval()
