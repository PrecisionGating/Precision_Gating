from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import imagenet_utils as utils


def apply_warmup(lr, base_lr, warmup_epochs, global_step):
  if warmup_epochs > 0:
    warmup_lr = (base_lr * tf.cast(global_step, tf.float32) / tf.cast(warmup_epochs, tf.float32))
    lr = tf.cond(global_step < warmup_epochs, lambda: warmup_lr, lambda: lr)

  return lr


def staircase(base_lr, batch_size, train_epochs, warmup_epochs, global_step):
  # Scale the learning rate linearly with the batch size. When the batch size
  # is 256, the learning rate should be 0.1.
  initial_lr = base_lr * batch_size / 256
  batches_per_epoch = utils._NUM_IMAGES['train'] / batch_size

  # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
  boundaries = [30, 60, 80, 90]
  lr_decays = [1, 0.1, 0.01, 1e-3, 1e-4]

  boundaries = [int(batches_per_epoch * epoch) for epoch in boundaries]
  values = [initial_lr * decay for decay in lr_decays]
  lr = tf.train.piecewise_constant(
      tf.cast(global_step, tf.int32), boundaries, values)

  return apply_warmup(lr, base_lr, warmup_epochs, global_step)

def linear_decay(base_lr, batch_size, train_epochs, warmup_epochs, global_step):
  batches_per_epoch = utils._NUM_IMAGES['train'] / batch_size
  end_lr = 1e-5
  total_steps = batches_per_epoch * train_epochs

  # Linearly decay the LR
  lr = tf.train.polynomial_decay(
      base_lr, tf.cast(global_step, tf.int32), total_steps,
      end_learning_rate=end_lr, power=1.0)

  return apply_warmup(lr, base_lr, warmup_epochs, global_step)
