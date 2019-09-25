from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
import numpy as np

# Gradient for the clip upper-bound in PACT
@tf.RegisterGradient('PactAlphaGrad')
def _pact_alpha_grad(op, grad):
  ''' Gradient for the upper-bound clipping in PACT
      y = min(x,a) elementwise
      dy/dx = 1 if x <= a else 0 elementwise
      dy/da = 1 if x >= a else 0 summed over x
  '''
  x = op.inputs[0]
  a = op.inputs[1]
  z = tf.zeros_like(grad)
  dydx = tf.where(x <= a, grad, z)
  dyda = tf.reduce_sum(tf.where(x >= a, grad, z))
  return dydx, dyda

def pact_clip(x, a):
  ''' Clipped ReLU with trainable clipping parameter alpha. It computes:
             0,  x <  0
        y =  x,  0 <= x < a
             a,  x >= a

      And defines the following gradients:
        dy/dx = ?
        dy/da = 0 if x < a else 1
  '''
  r = tf.nn.relu(x)
  g = tf.get_default_graph()
  with g.gradient_override_map({'Minimum': 'PactAlphaGrad'}):
    return tf.minimum(r, a)

def pact_relu(x, init_value=6.0, return_threshold=False, name=None):
  ''' Replaces tf.nn.relu for PACT '''
  # Trainable upper bound for activtaions
  with tf.variable_scope(name, default_name='pact') as scope:
    alpha = tf.get_variable('pact_alpha', shape=(), dtype=x.dtype,
                            initializer=tf.constant_initializer(init_value))

    tf.summary.scalar(scope.name+'/alpha', alpha)

    if not return_threshold:
      return pact_clip(x, alpha)
    else:
      return pact_clip(x, alpha), alpha


if __name__ == '__main__':
  _x = np.array([-2, -0.5, 0.5, 2.0, 3.0, 4.0])
  x = tf.Variable(_x, dtype=tf.float32)
  a = tf.Variable(2.5, dtype=tf.float32)

  y1 = tf.nn.relu(x)
  y2 = relu_pact(x,a)

  g1 = tf.gradients(y1, x)
  g2 = tf.gradients(y2, x)
  g3 = tf.gradients(y2, a)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    _y1 = y1.eval(session=sess)
    _y2 = y2.eval(session=sess)
    _g1 = [g.eval(session=sess) for g in g1]
    _g2 = [g.eval(session=sess) for g in g2]
    _g3 = [g.eval(session=sess) for g in g3]

  print('x')
  print(_x)
  print('\ny1')
  print(_y1)
  print(_g1[0])
  print('\ny2')
  print(_y2)
  print(_g2[0])
  print(_g3[0])
