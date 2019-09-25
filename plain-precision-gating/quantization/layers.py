from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, re
import tensorflow as tf
import numpy as np

import quantization.quantize as q
import quantization.util_layers as util
import quantization.pact as pact
import utils.json as my_json

BN_MOMENTUM = 0.99
BN_EPSILON = 1e-5

# JSON file name
READ_CONFIG_JSON = None
WRITE_CONFIG_JSON = None

def set_read_json(fname):
  global READ_CONFIG_JSON
  READ_CONFIG_JSON = fname
def set_write_json(fname):
  global WRITE_CONFIG_JSON
  WRITE_CONFIG_JSON = fname


def get_kern_initializer(wbits):
  if wbits == 1:
    return tf.contrib.layers.xavier_initializer()
  elif wbits == 2:
    return tf.truncated_normal_initializer(stddev=0.5)
  else:
    return tf.variance_scaling_initializer()


#-------------------------------------------------------------------------
# Quantized depthwise conv2d
#-------------------------------------------------------------------------
def depthwise_conv2d(inputs, kernel_size, **kwargs):
    return predictive_conv2d(inputs, 0, kernel_size, depthwise=True, **kwargs)

#-------------------------------------------------------------------------
# Quantized conv2d
#-------------------------------------------------------------------------
def conv2d(inputs, filters, kernel_size,
           strides=1, padding='SAME', depthwise=False,
           kern_initializer=None,
           bias_initializer=None,
           wbits=0, abits=0, stochastic=False,
           data_format='channels_first',
           use_bias=False,
           first_layer=False,
           clip_weights=False,
           summary=False,
           _dict=None,
           name=None):
  """conv2d returns a 2d convolution layer with full stride."""
  # Handle read/write json file
  if WRITE_CONFIG_JSON is not None:
    config = {'wbits': wbits, 'abits': abits}
    my_json.write_or_update_json(WRITE_CONFIG_JSON, name, config)
  if READ_CONFIG_JSON is not None:
    config = my_json.access_json(READ_CONFIG_JSON, name)
    wbits = config['wbits']
    abits = config['abits']

  if data_format == 'channels_last':
    in_channels = inputs.shape.as_list()[-1]
    strides = [1, strides, strides, 1]
    conv2d_data_format = 'NHWC'
  elif data_format == 'channels_first':
    in_channels = inputs.shape.as_list()[1]
    strides = [1, 1, strides, strides]
    conv2d_data_format = 'NCHW'
  else:
    raise ValueError('Unrecognized data format %s' % data_format)

  if not depthwise:
    print('* Quantized conv2d: %sw/%sa (%d -> %d channels)' %
            (q.b2s(wbits), q.b2s(abits), in_channels, filters))
  else:
    print('* Quantized conv2d: %sw/%sa (%d channels depthwise)' %
            (q.b2s(wbits), q.b2s(abits), in_channels))


  if kern_initializer == None:
    kern_initializer = get_kern_initializer(wbits)
  if bias_initializer == None:
    bias_initializer = tf.zeros_initializer()

  with tf.variable_scope(name, default_name='conv2d') as scope:
    Wshape = (kernel_size, kernel_size, in_channels, (1 if depthwise else filters))

    constraint = (lambda x: tf.clip_by_value(x,-1,1)) \
                 if clip_weights else None
    W = tf.get_variable(
        "weights", Wshape, dtype=inputs.dtype,
        initializer=kern_initializer,
        constraint=constraint)

    if summary:
      _variable_summary(W)

    # Quantize the weight
    Wb = q.quantize_w(W, wbits, stochastic, name="W_b")
    if wbits == 1:
      var_w = 1.0 / (kernel_size**2 * in_channels)
      Wb = Wb * tf.constant(
          np.sqrt(var_w), dtype=Wb.dtype, name="Wb_var_scale")

    # Quantize the input activations
    Ab = q.quantize_a(inputs, abits, stochastic, name='A_b')

    # Perform conv
    if depthwise:
      h = tf.nn.depthwise_conv2d(
          Ab, Wb, strides=strides, padding=padding,
          data_format=conv2d_data_format, name=name)
    else:
      h = tf.nn.conv2d(
          Ab, Wb, strides=strides, padding=padding,
          data_format=conv2d_data_format, name=name)

    if summary:
      _variable_summary(h)

    if use_bias:
      b = tf.get_variable(
          "bias", (out_channels), dtype=inputs.dtype,
          initializer=bias_initializer)
      h = tf.nn.bias_add(h, b)

    # Store activations
    if _dict is not None:
      _dict[scope.name+'/out'] = h
      _dict[scope.name+'/w'] = W
      _dict[scope.name+'/w_q'] = Wb
      _dict[scope.name+'/in'] = inputs
      _dict[scope.name+'/in_q'] = Ab

    return h

#-------------------------------------------------------------------------
# Quantized fc
#-------------------------------------------------------------------------
def fc(inputs, units,
       kern_initializer=None,
       bias_initializer=None,
       wbits=0, abits=0, stochastic=False,
       use_bias=False,
       first_layer=False,
       clip_weights=False,
       summary=False,
       _dict=None,
       name=None):
  """dense layer"""
  # Handle read/write json file
  if WRITE_CONFIG_JSON is not None:
    config = {'wbits': wbits, 'abits': abits}
    my_json.write_or_update_json(WRITE_CONFIG_JSON, name, config)
  if READ_CONFIG_JSON is not None:
    config = my_json.access_json(READ_CONFIG_JSON, name)
    wbits = config['wbits']
    abits = config['abits']

  in_units = inputs.shape.as_list()[-1]

  print('* Quantized fc: %sw/%sa (%d -> %d channels)' %
          (q.b2s(wbits), q.b2s(abits), in_units, units))

  if kern_initializer == None:
    kern_initializer = get_kern_initializer(wbits)
  if bias_initializer == None:
    bias_initializer = tf.zeros_initializer()

  with tf.variable_scope(name, default_name='fc') as scope:

    Wshape=(in_units, units)
    constraint = (lambda x: tf.clip_by_value(x,-1,1)) \
                 if clip_weights else None
    W = tf.get_variable(
        "weights", Wshape, dtype=inputs.dtype,
        initializer=kern_initializer,
        constraint=constraint)

    if summary:
      _variable_summary(W)

    # Quantize the weight
    Wb = q.quantize_w(W, wbits, stochastic, name="W_b")

    if wbits == 1:
      var_w = 1.0 / in_units
      Wb = Wb * tf.constant(
          np.sqrt(var_w), dtype=Wb.dtype, name="Wb_var_scale")

    # Quantize the input activations
    Ab = q.quantize_a(inputs, abits, stochastic, name='A_b')

    # Perform matmul
    h = tf.matmul(Ab, Wb)

    if summary:
      _variable_summary(h)

    if use_bias:
      b = tf.get_variable(
          "bias", (units), dtype=inputs.dtype,
          initializer=bias_initializer)
      h = tf.nn.bias_add(h, b)

    # Store activations before quantization
    if _dict is not None:
      _dict[scope.name+'/out'] = h
      _dict[scope.name+'/w'] = W
      _dict[scope.name+'/w_q'] = Wb
      _dict[scope.name+'/in'] = inputs
      _dict[scope.name+'/in_q'] = Ab

    return h


#-------------------------------------------------------------------------
# Other layers
#-------------------------------------------------------------------------
batch_norm = util.batch_norm
batch_norm_relu = util.batch_norm_relu
nn_conv2d = util.nn_conv2d
pact_relu = pact.pact_relu



#-------------------------------------------------------------------------
# Predictive Convolution Layer
#-------------------------------------------------------------------------
def predictive_conv2d(inputs, 
                      filters, 
                      kernel_size,
                      strides=1, 
                      padding='SAME', 
                      depthwise=False,
                      kern_initializer=None,
                      bias_initializer=None,
                      wbits=0, 
                      abits=0, 
                      stochastic=False,
                      data_format='channels_first',
                      use_bias=False,
                      first_layer=False,
                      clip_weights=False,
                      summary=False,
                      _dict=None,
                      name=None,
                      trunc_to_bits=0,
                      is_training=True):

  # get the number of input channels
  if data_format == 'channels_last':
    in_channels = inputs.shape.as_list()[-1]
    strides = [1, strides, strides, 1]
    conv2d_data_format = 'NHWC'
  elif data_format == 'channels_first':
    in_channels = inputs.shape.as_list()[1]
    strides = [1, 1, strides, strides]
    conv2d_data_format = 'NCHW'
  else:
    raise ValueError('Unrecognized data format %s' % data_format)

  # print network info
  if not depthwise:
    print('* Truncated predictive conv2d: %sw/%sa/%strunc_bits (%d -> %d channels)' %
            (q.b2s(wbits), q.b2s(abits), q.b2s(trunc_to_bits), in_channels, filters))
  else:
    print('* Depthwise conv2d: %sw/%sa/(no truncation) (%d channels depthwise)' %
            (q.b2s(wbits), q.b2s(abits), in_channels))

  if kern_initializer is None:
    kern_initializer = get_kern_initializer(wbits)
  if bias_initializer is None:
    bias_initializer = tf.zeros_initializer()


  inputs = q.quantize_a(inputs, bits=abits, name="full_precision_inputs")

  # Initialize the weights and bias for the shared conv2d
  # All operations and variables will be under the variable scope
  with tf.variable_scope(name, default_name='predictive_conv2d') as scope:

    Wshape = (kernel_size, kernel_size, in_channels, (1 if depthwise else filters))
    constraint = (lambda x: tf.clip_by_value(x,-1,1)) \
                 if clip_weights else None
    W = tf.get_variable(
        "weights", Wshape, dtype=inputs.dtype,
        initializer=kern_initializer,
        constraint=constraint)
    b = tf.get_variable("bias", 
                        (out_channels), 
                        dtype=inputs.dtype, 
                        initializer=bias_initializer) \
        if use_bias else None

    if depthwise:
      outputs = tf.nn.depthwise_conv2d(
                inputs, W, strides=strides, padding=padding,
                data_format=conv2d_data_format, name=name+'higher_bit_conv')
    else:
      higher_bit_part_inputs = q.quantize_a(inputs,
                                            bits=abits,
                                            trunc_to_bits=trunc_to_bits,
                                            name=name+"truncate_inputs")

      h_Hb = tf.nn.conv2d(
             higher_bit_part_inputs, W, strides=strides, padding=padding,
             data_format=conv2d_data_format, name=name+'higher_bit_conv')

      h_full = tf.nn.conv2d(
               inputs, W, strides=strides, padding=padding,
               data_format=conv2d_data_format, name=name+'higher_bit_conv')

      constraint = lambda x: tf.clip_by_value(x, -20, 20)
      delta = tf.get_variable(name=name+"training_threshold",
                              shape=[1, filters, 1, 1], 
                              dtype=inputs.dtype,
                              initializer=tf.zeros_initializer(),
                              constraint=constraint)

      forward_mask = tf.where(tf.greater(h_Hb, delta),
                              tf.ones_like(h_Hb),
                              tf.zeros_like(h_Hb),
                              name=name+"forward_mask")

      backward_mask = tf.sigmoid(5 * (h_Hb - delta), name=name+"backward_mask")

      mask = backward_mask + tf.stop_gradient(forward_mask - backward_mask)

      outputs = h_Hb + mask * (h_full - h_Hb)

      tf.add_to_collection(tf.GraphKeys.METRIC_VARIABLES,
                         tf.identity(tf.reduce_sum(forward_mask),
                                     name=name+'num_of_full_precision_conv'))

      tf.add_to_collection(tf.GraphKeys.METRIC_VARIABLES,
                         tf.identity(tf.reduce_sum(tf.ones_like(forward_mask)),
                                     name=name+'total_num_of_conv'))

    return outputs




def _variable_summary(x):
  name = re.sub(':', '_', x.name)
  mean, var = tf.nn.moments(x, axes=list(range(x.shape.ndims)))
  tf.summary.scalar(name + '/mean', mean)
  tf.summary.scalar(name + '/var', var)
