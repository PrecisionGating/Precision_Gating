from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import inspect

import quantization.layers as layers


def get_func_kwargs(func):
  args, varargs, keywords, defaults = inspect.getargspec(func)
  return args[-len(defaults):]

def num_channels(tensor, data_format):
  shape = tensor.shape.as_list()
  if data_format == "channels_last":
    return shape[-1]
  elif data_format == "channels_first":
    return shape[1]
  else:
    raise ValueError ('Unrecognized data_format %s' % data_format)

def get_channel_axis(data_format):
  if data_format == 'channels_last':
    return 3
  elif data_format == 'channels_first':
    return 1
  else:
    raise ValueError ('Unrecognized data_format %s' % data_format)

def total_weight_size():
  tvars = tf.trainable_variables()
  wt_size = 0
  for v in tvars:
    v.shape.assert_is_fully_defined()
    wt_size += np.prod(v.shape.as_list())
  return wt_size

def reshape_like(x, target):
  """ Reshape x to the shape of the target, excluding batch dimension """
  target_shape = target.shape.as_list()
  target_shape[0] = -1
  return tf.reshape(x, target_shape)


#-------------------------------------------------------------------------
# Explicit padding for conv layers
#-------------------------------------------------------------------------
def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size,
                         strides=1, padding='SAME',
                         data_format='channels_first',
                         first_layer=False,
                         name='default_name',
                         is_training=True,
                         **kwargs):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  if first_layer:
      print("This layer is not truncated or quantized!")
      return layers.conv2d(
          inputs, filters, kernel_size,
          strides=strides,
          padding=('SAME' if strides == 1 else 'VALID'),
          data_format=data_format,
          )
  else:
      return layers.predictive_conv2d(inputs, filters, kernel_size,
                                      strides=strides,
                                      padding=('SAME' if strides == 1 else 'VALID'),
                                      wbits=kwargs['wbits'],
                                      abits=kwargs['abits'],
                                      data_format=data_format,
                                      first_layer=first_layer,
                                      name=(name+'_conv2d_fixed_padding_'),
                                      is_training=is_training,
                                      trunc_to_bits=kwargs['trunc_to_bits'])



#-------------------------------------------------------------------------
# A resnet block layer
#-------------------------------------------------------------------------
def block_layer(inputs, filters, block_fn, blocks, strides,
                is_training, name, data_format,
                use_pact=False, **kwargs):
  """Creates one layer of blocks.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Build the block layer
  with tf.variable_scope(name) as scope:
    # Only the first block per block_layer uses projection_shortcut and strides
    with tf.variable_scope('block_1') as scope:
      inputs = block_fn(inputs, filters, is_training=is_training, projection=True,
                        strides=strides, data_format=data_format,
                        use_pact=use_pact, **kwargs)

    for i in range(1, blocks):
      with tf.variable_scope('block_%d' % (i+1)) as scope:
        inputs = block_fn(inputs, filters, is_training=is_training,
                          strides=1, data_format=data_format, projection=False,
                          use_pact=use_pact, **kwargs)

  return tf.identity(inputs, name)
