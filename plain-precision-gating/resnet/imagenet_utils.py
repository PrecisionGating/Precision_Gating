from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys

import numpy as np
import tensorflow as tf

import imagenet_preprocessing
from pkg_resources import parse_version

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_NUM_VALIDATION_FILES = 128
_SHUFFLE_BUFFER = 10000


def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(_NUM_VALIDATION_FILES)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, tf.one_hot(label, _LABEL_CLASSES)


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, dtype=tf.float32):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.
    dtype: Data type to use for images/features.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset = dataset.take(total_batches * batch_size)

  elif is_training and parse_version(tf.__version__) < parse_version('1.8.0'):
    # For versions less than 1.8 we have to make the total divisible by
    # batch size or get a shape error on the last batch
    assert(examples_per_epoch is not None)
    total_examples = num_epochs * examples_per_epoch
    if num_gpus:
      total_batches = total_examples // batch_size // num_gpus * num_gpus
    else:
      total_batches = total_examples // batch_size
    dataset = dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  if parse_version(tf.__version__) < parse_version('1.8.0'):
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=1))
  else:
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=True))     # tf 1.8 or later

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  if parse_version(tf.__version__) < parse_version('1.8.0'):
    dataset = dataset.prefetch(buffer_size=batch_size)
  else:
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None,
             dtype=tf.float32):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=10))

  return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None,
      dtype=dtype
  )

#----------------------------------------------------
# RZ: Old functions
#----------------------------------------------------
#def record_parser(value, is_training):
#  """Parse an ImageNet record from `value`."""
#  keys_to_features = {
#      'image/encoded':
#          tf.FixedLenFeature((), tf.string, default_value=''),
#      'image/format':
#          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
#      'image/class/label':
#          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
#      'image/class/text':
#          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#      'image/object/bbox/xmin':
#          tf.VarLenFeature(dtype=tf.float32),
#      'image/object/bbox/ymin':
#          tf.VarLenFeature(dtype=tf.float32),
#      'image/object/bbox/xmax':
#          tf.VarLenFeature(dtype=tf.float32),
#      'image/object/bbox/ymax':
#          tf.VarLenFeature(dtype=tf.float32),
#      'image/object/class/label':
#          tf.VarLenFeature(dtype=tf.int64),
#  }
#
#  parsed = tf.parse_single_example(value, keys_to_features)
#
#  image = tf.image.decode_image(
#      tf.reshape(parsed['image/encoded'], shape=[]),
#      _NUM_CHANNELS)
#  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#
#  image = imagenet_preprocessing.preprocess_image(
#      image=image,
#      output_height=_DEFAULT_IMAGE_SIZE,
#      output_width=_DEFAULT_IMAGE_SIZE,
#      is_training=is_training)
#
#  label = tf.cast(
#      tf.reshape(parsed['image/class/label'], shape=[]),
#      dtype=tf.int32)
#
#  return image, tf.one_hot(label, _LABEL_CLASSES)


#def input_fn(is_training, data_dir, batch_size, num_epochs=1):
#  """Input function which provides batches for train or eval."""
#  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
#
#  if is_training:
#    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)
#
#  dataset = dataset.flat_map(tf.data.TFRecordDataset)
#  dataset = dataset.map(lambda value: record_parser(value, is_training),
#                        num_parallel_calls=5)
#  dataset = dataset.prefetch(batch_size)
#
#  if is_training:
#    # When choosing shuffle buffer sizes, larger sizes result in better
#    # randomness, while smaller sizes have better performance.
#    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)
#
#  # We call repeat after shuffling, rather than before, to prevent separate
#  # epochs from blending together.
#  dataset = dataset.repeat(num_epochs)
#  dataset = dataset.batch(batch_size)
#
#  iterator = dataset.make_one_shot_iterator()
#  images, labels = iterator.get_next()
#  return images, labels
