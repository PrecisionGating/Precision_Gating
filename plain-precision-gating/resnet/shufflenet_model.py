from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import quantization.layers as layers
from quantization.layers import depthwise_conv2d
from common import *

#-------------------------------------------------------------------------
# ShuffleNetV2
#-------------------------------------------------------------------------
def shufflenet_v2_block(inputs, filters, is_training, strides, data_format,
                        projection=False, use_pact=False, **kwargs):

  channel_axis = get_channel_axis(data_format)
  channels = num_channels(inputs, data_format)

  cell_filters = filters // 2
  shortcut_filters = filters - cell_filters

  if strides == 1:
    assert (channels == filters)

    # Split inputs
    shortcut, inputs = tf.split(
        inputs, [shortcut_filters, cell_filters],
        axis=channel_axis, name='channel_split')
  else:
    # Shortcut path, expand channels after the depthwise conv
    shortcut = depthwise_conv2d(
        inputs, 3, strides=strides, data_format=data_format,
        name='shortcut_dw', is_training=is_training, **kwargs)
    shortcut = layers.batch_norm(shortcut, is_training, data_format)

    shortcut = conv2d_fixed_padding(
        shortcut, shortcut_filters, 1, strides=1, data_format=data_format,
        name='shortcut_conv', is_training=is_training, **kwargs)
    shortcut = layers.batch_norm_relu(shortcut, is_training, data_format, use_pact)

  # Main path, expand channels immediately
  inputs = conv2d_fixed_padding(
      inputs, cell_filters, 1, strides=1, data_format=data_format,
      name='conv1', is_training=is_training, **kwargs)
  inputs = layers.batch_norm_relu(inputs, is_training, data_format, use_pact)

  inputs = depthwise_conv2d(
      inputs, 3, strides=strides, data_format=data_format,
      name='conv2', is_training=is_training, **kwargs)
  inputs = layers.batch_norm(inputs, is_training, data_format)

  inputs = conv2d_fixed_padding(
      inputs, cell_filters, 1, strides=1, data_format=data_format,
      name='conv3', is_training=is_training, **kwargs)
  inputs = layers.batch_norm_relu(inputs, is_training, data_format, use_pact)

  # Join the two paths and simultaneously perform channel shuffle
  # We do this by stacking the two paths in an inner dimension
  shape = inputs.shape.as_list()
  shape[channel_axis] = 2*shape[channel_axis]
  inputs = tf.stack([shortcut, inputs], axis=channel_axis+1, name='channel_join')
  inputs = tf.reshape(inputs, shape)

  if use_pact:
    return layers.pact_relu(inputs)
  else:
    return inputs


#-------------------------------------------------------------------------
# ImageNet shufflenet generator
#-------------------------------------------------------------------------
def imagenet_shufflenet_generator(num_classes, block_fn, num_blocks, block_filters,
                                  version=1, data_format=None, **kwargs):
  """Generator for ImageNet ResNet v2 models.

  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  print("imagenet_shufflenet_generator")
  print("  version     = %d" % version)
  print("  num_blocks    =", num_blocks)
  print("  block_filters =", block_filters)
  print("  num_classes = %d" % num_classes)
  print("  data_format = %s" % data_format)
  for k,v in kwargs.items():
    print(" ", k, "=", v)

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=24, kernel_size=3, strides=2,
        data_format=data_format, first_layer=True, name='first_layer', **kwargs)
    inputs = layers.batch_norm_relu(inputs, is_training, data_format,
                                    use_pact=kwargs['use_pact'])
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs, block_filters[0], block_fn, num_blocks[0],
        strides=2, is_training=is_training, name='block_layer1',
        data_format=data_format, **kwargs)
    inputs = block_layer(
        inputs, block_filters[1], block_fn, num_blocks[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format, **kwargs)
    inputs = block_layer(
        inputs, block_filters[2], block_fn, num_blocks[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format, **kwargs)

    if len(block_filters) > 3:
      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=block_filters[3], kernel_size=1, strides=1,
          data_format=data_format, first_layer=True, name='last_layer', **kwargs)
      inputs = layers.batch_norm_relu(inputs, is_training, data_format,
                                      use_pact=kwargs['use_pact'])
      inputs = tf.identity(inputs, 'final_conv')

    # feature maps will be 7x7 here
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=7, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    s = inputs.shape
    inputs = tf.reshape(inputs, [-1, s[1]*s[2]*s[3]])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')

    print("\nTotal weight size = %8.2e" % total_weight_size())

    return inputs

  return model
