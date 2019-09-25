# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys

import numpy as np
import tensorflow as tf

import imagenet_utils as utils
from shufflenet_model import *
from common import *
from optimize import build_train_or_eval_graph
import lr_schedule

import time

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/fast_disk/imagenet',
                    help='The directory where the ImageNet input data is stored.')

parser.add_argument('--model_dir', type=str, default='./imagenet_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--model', '-m', type=str, required=True,
                    help="Model name in the form of '<name>-<variant>'. Choices are "
                         "'resnet-18/34', "
                         "'bottleneck-50/101/152/200', "
                         "'shufflenetv2-0.5x/1x/1.5x/2x'")

parser.add_argument('--version', '-v', type=int, default=1,
                    help="resnet v1 or v2")

parser.add_argument('--wide', type=float, default=1.,
                    help='Channel width multiplier.')

parser.add_argument('--train_epochs', type=int, default=100,
                    help='The number of epochs to use for training.')

parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='Epochs of warmup where learning rate rises linearly to the intial value.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for training and evaluation.')

parser.add_argument('--log_every_n_iters', type=int, default=100,
                    help='Logging frequency.')

parser.add_argument('--test_only', action='store_true',
                    help='Only run the evaluation.')

parser.add_argument('--use_pact', action='store_true')
parser.add_argument('--wbits', type=int, default=0, help="Weight bits")
parser.add_argument('--abits', type=int, default=0, help="Activation bits")

parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use")
parser.add_argument('--which_gpus', type=str, default='0', help="Which GPUs to use")

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

parser.add_argument('--trunc_to_bits', type=int, default=0, help="Number of bits used to do conv prediction.")
parser.add_argument('--gating_target', type=float, default=0., help="Target gating value")

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4
_GATING_LOSS = 1e-3

res_net_params = {
  '18': {'blocks': [2, 2, 2, 2], 'filters': [64, 128, 256, 512]},
  '34': {'blocks': [3, 4, 6, 3], 'filters': [64, 128, 256, 512]}
}

bottleneck_params = {
  '50':  {'blocks': [3,  4,  6, 3], 'filters': [64, 128, 256, 512]},
  '101': {'blocks': [3,  4, 23, 3], 'filters': [64, 128, 256, 512]},
  '152': {'blocks': [3,  8, 36, 3], 'filters': [64, 128, 256, 512]},
  '200': {'blocks': [3, 24, 36, 3], 'filters': [64, 128, 256, 512]}
}

# This is a bit confusing, but ShuffleNetV1 has channels per layer
# dependent on the group number, and can be scaled with a width multiplier
# But ShuffleNetV2 has the channels per layer dependent on the 
# width multiplier only (group always = 2)
ShuffleNetV1_params = {
  'g1': {'blocks': [4,8,4], 'filters': [144, 288,  576], 'groups': 1},
  'g2': {'blocks': [4,8,4], 'filters': [200, 400,  800], 'groups': 2},
  'g3': {'blocks': [4,8,4], 'filters': [240, 480,  960], 'groups': 3},
  'g4': {'blocks': [4,8,4], 'filters': [272, 544, 1088], 'groups': 4},
  'g8': {'blocks': [4,8,4], 'filters': [384, 768, 1536], 'groups': 8},
}

# Last elem in filters is the Conv5 width
ShuffleNetV2_params = {
  'x0.25': {'blocks': [4,8,4], 'filters': [24,  48,   96,  512]},
  'x0.5':  {'blocks': [4,8,4], 'filters': [48,  96,  192, 1024]},
  'x1':    {'blocks': [4,8,4], 'filters': [116, 232, 464, 1024]},
  'x1.5':  {'blocks': [4,8,4], 'filters': [176, 352, 704, 1024]},
  'x2':    {'blocks': [4,8,4], 'filters': [244, 488, 976, 2048]},
}

PARAMS = {
  'resnet': res_net_params,
  'bottleneck': bottleneck_params,
  'shufflenetv1': ShuffleNetV1_params,
  'shufflenetv2': ShuffleNetV2_params
}

def imagenet_model_generator(num_classes, params):
  """Returns an ImageNet model based on the type (resnet, bottleneck, shuffle),
     size, width_multiplier, and number of output classes."""

  model_name = params['model']
  version = params['version']

  # Parse the model name, format='<model>-<variant>'
  toks = model_name.split('-')
  assert(len(toks) == 2)
  model_name, variant = toks[0], toks[1]
  model_name = model_name.lower()

  # Define each model here
  if model_name == 'resnet':
    _block_fn = building_block_v1 if version==1 else building_block_v2
    _generator = imagenet_resnet_generator
    _net_params = PARAMS['resnet']

  elif model_name == 'bottleneck':
    _block_fn = bottleneck_block_v1 if version==1 else bottleneck_block_v2
    _generator = imagenet_resnet_generator
    _net_params = PARAMS['bottleneck']

  elif model_name == 'shufflenetv1':
    _block_fn = shufflenet_v1_block
    _generator = imagenet_shufflenet_generator
    _net_params = PARAMS['shufflenetv1']

  elif model_name == 'shufflenetv2':
    _block_fn = shufflenet_v2_block
    _generator = imagenet_shufflenet_generator
    _net_params = PARAMS['shufflenetv2']

  else:
    raise ValueError('Unrecognized model name %s' % name)

  # Get the hyperparameters for the model variant
  if variant not in _net_params:
    raise ValueError('Not a valid %s variant: %s' % (model_name, variant))
  _net_params = _net_params[variant]

  _num_blocks = _net_params['blocks']
  _filters = _net_params['filters']
  _filters = [int(f*params['wide']) for f in _filters]

  ## kwargs for the generator
  kwargs = {k: v for k,v in params.items() if v is not None \
            and k in \
              get_func_kwargs(_generator) + \
              get_func_kwargs(_block_fn) + \
              ['wbits', 'abits', 'data_format', 'trunc_to_bits']}

  if model_name == 'shufflenetv1':
    kwargs['groups'] = _net_params['groups']

  return _generator(
      num_classes, _block_fn, _num_blocks, _filters, **kwargs)


def imagenet_model_fn(features, labels, mode, params):

  tf.summary.image('images', features, max_outputs=6)

  network = imagenet_model_generator(utils._LABEL_CLASSES, params)

  # Predict mode, return EstimatorSpec containing predictions
  # See www.tensorflow.org/get_started/custom_estimators
  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = network(features, False)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Code below is either for Train mode or Eval mode.
  # Train mode: return EstimatorSpec containing loss and train_op
  # Eval  mode: return EstimatorSpec containing loss
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    # Define the learning rate schedule and log the lr tensor
    learning_rate = lr_schedule.linear_decay(
            0.5, params['batch_size'], params['train_epochs'],
            0, global_step)
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    # Create optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)
  else:
    optimizer = None

  # Get loss and train op
  logits, loss, train_op = build_train_or_eval_graph(
          network, features, labels, optimizer,
          is_training=(mode == tf.estimator.ModeKeys.TRAIN),
          num_gpus=params['num_gpus'],
          weight_decay=4e-5,
          gating_loss_factor=_GATING_LOSS,
          delta_T=FLAGS.gating_target)

  # Compute predictions and accuracy
  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  # add a new metric -> the percentage of full precision convolutions
  percent = tf.add_n([tf.reduce_sum(v) for v in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES) 
                      if 'num_of_full_precision_conv' in v.name]) / \
            tf.add_n([tf.reduce_sum(v) for v in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES) 
                      if 'total_num_of_conv' in v.name])
  tf.identity(percent, name='avg_percent')
  tf.summary.scalar('avg_percent', percent)
  percent_for_report = tf.metrics.mean(percent)
  tf.identity(percent_for_report[1], name='percent_for_report')
  tf.summary.scalar('percent_for_report', percent_for_report[1])

  # register cross entropy loss
  cross_entropy_loss = tf.add_n([v for v in tf.get_collection(tf.GraphKeys.LOSSES) 
                                 if 'cross_entropy' in v.name])
  tf.identity(cross_entropy_loss, name='cross_entropy_loss')
  tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

  # register gating loss
  gating_loss = tf.add_n([v for v in tf.get_collection(tf.GraphKeys.LOSSES) 
                                 if 'gating_loss' in v.name])
  tf.identity(gating_loss, name='gating_loss_')
  tf.summary.scalar('gating_loss_', gating_loss)

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy, 'percent': percent_for_report}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.which_gpus

  if not os.path.isdir(FLAGS.data_dir):
    raise ValueError('Could not find data_dir %s' % FLAGS.data_dir)

  # TF estimator API requires cifar10_model_fn to pass arguments using a
  # 'params' dict. So we put all FLAGS args into 'params'
  params = {k: v for k,v in FLAGS._get_kwargs() if v is not None} 

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  resnet_classifier = tf.estimator.Estimator(
      model_fn=imagenet_model_fn, model_dir=FLAGS.model_dir, config=run_config, params=params)

  #-------------------------------#
  # Test only execution           #
  #-------------------------------#
  if FLAGS.test_only == True:
    print("@ Start to register time.")
    ttt = time.time()
    # Evaluate the model and print results
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: utils.input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print("\nTest results after step %d:" % eval_results['global_step'])
    print(" Val loss: %f" % eval_results['loss'])
    print(" Val acc : %6.3f" % eval_results['accuracy'])
    print(" Val percent of full precision conv : %6.3f" % eval_results['percent'])
    print(" Time elapsed : %6.3f" % (time.time()-ttt))
    sys.exit(0)

  #-------------------------------#
  # Train/eval cycles             #
  #-------------------------------#
  epochs_per_eval = min(FLAGS.train_epochs, FLAGS.epochs_per_eval)

  for i in range(FLAGS.train_epochs // epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy_loss',
        'gating_loss': 'gating_loss_',
        'train_accuracy': 'train_accuracy',
        'percent': 'avg_percent'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_iters)

    print('===== Starting a training cycle =====')
    resnet_classifier.train(
        input_fn=lambda: utils.input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])
    print("")

    print('===== Starting to evaluate =====')
    # Evaluate the model and print results
    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: utils.input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print("\nTest results after step %d:" % eval_results['global_step'])
    print(" Val loss: %f" % eval_results['loss'])
    print(" Val acc : %6.3f" % eval_results['accuracy'])
    print(" Val percent of full precision conv : %6.3f" % eval_results['percent'])
    print("")


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
