from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, re
import tensorflow as tf
import numpy as np

import quantization.quantize as q
from quantization.pact import pact_relu

#-------------------------------------------------------------------------
# Batch norm methods
#-------------------------------------------------------------------------
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, is_training, data_format):
  assert(data_format == 'channels_first' or data_format == 'channels_last')
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)

def batch_norm_relu(inputs, is_training, data_format, use_pact=False):
  """Performs a batch normalization followed by a ReLU."""
  assert(data_format == 'channels_first' or data_format == 'channels_last')
  inputs = batch_norm(inputs, is_training, data_format)
  if use_pact:
    inputs = pact_relu(inputs)
  else:
    inputs = tf.nn.relu(inputs)
  return inputs

#-------------------------------------------------------------------------
# Wrapper for tf.nn.conv2d
#-------------------------------------------------------------------------
def nn_conv2d(inputs, weights, stride=1, padding='SAME', data_format='channels_last', name=None):
  """Simpler interface for tf.nn.conv2d"""
  if data_format == 'channels_last':
    strides = [1, stride, stride, 1]
    conv2d_data_format = 'NHWC'
  elif data_format == 'channels_first':
    strides = [1, 1, stride, stride]
    conv2d_data_format = 'NCHW'
  else:
    raise ValueError('Unrecognized data format %s' % data_format)

  return tf.nn.conv2d(inputs, weights, strides, padding, data_format=conv2d_data_format, name=name)
