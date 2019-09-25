from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
import numpy as np

_W_METHOD = None
_A_METHOD = None


_GRAD_DEFINED = False
if not _GRAD_DEFINED:
  @tf.RegisterGradient("IdentityMaxMinGrad")
  def _identity_max_min_grad(op, grad):
    return grad, None

def tf_log2(x):
  return tf.log(x) / tf.cast(tf.log(2.0), x.dtype)
def tf_pow2(x):
  return tf.pow(tf.cast(2.0, x.dtype), x)

def b2s(bits):
  if type(bits) is int:
    return str(bits)
  elif type(bits) is list:
    return "%4.1f" % np.mean(bits)
  else:
    raise ValueError('Cannot print bits when its type is %s' % str(type(bits)))

#--------------------------------------------------------------------
# Binarization function
#--------------------------------------------------------------------
def binarize(x, stochastic=False, name=None):

  g = tf.get_default_graph()
  with g.gradient_override_map({'Minimum': 'IdentityMaxMinGrad', 'Maximum': 'IdentityMaxMinGrad'}):
    xc = tf.clip_by_value(x, -1, 1)

  xb = (xc+1.)/2.

  g = tf.get_default_graph()
  with g.gradient_override_map({'Round': 'Identity', 'Floor': 'Identity'}):
    # Round to {0,1}
    if stochastic:
      r = tf.random_uniform(tf.shape(xb), 0,1)
      xb = tf.floor(xb+r)
    else:
      xb = tf.round(xb)

  # Rescale back to {-1, 1}
  xb = xb*2. - 1.
  return tf.identity(xb, name=name) if name else xb

#--------------------------------------------------------------------
# Convert a value to fixed point
#--------------------------------------------------------------------
def round_to_bits(x, bits=0, stochastic=False, trunc_to_bits=0):

  if bits == 0:
    return x
  if type(bits) is list:
    bits = np.array(bits)

  bitshift = 2**bits - 1

  # Override gradient of round/floor to identity
  g = tf.get_default_graph()
  with g.gradient_override_map({'Round': 'Identity', 'Floor': 'Identity'}):
    if stochastic:
      r = tf.random_uniform(Q.shape, 0,1)
      return tf.floor(x*bitshift + r) / bitshift
    else:
      truncation = tf.round(x*bitshift)
      if(trunc_to_bits==0):
        return truncation / bitshift
      else:
        remove_bits = bits - trunc_to_bits
        remove_shift = 2**remove_bits
        return tf.floor(truncation / remove_shift) * remove_shift / bitshift

#--------------------------------------------------------------------
# Quantization function for weights and activations
#--------------------------------------------------------------------
def quantize_w(x, bits=0, stochastic=False, name=None):

  if bits == 0:
    return x
  if bits == 1:
    return binarize(x, stochastic=stochastic, name=name)

  print('  * quantize_w: bits=%s' % b2s(bits))

  if _W_METHOD == 'dorefa':
    x = tf.nn.tanh(x)


  sign = tf.stop_gradient(tf.sign(x))
  scaling = tf.reduce_max(tf.abs(x))
  scaling = tf.stop_gradient(scaling)

  g = tf.get_default_graph()
  with g.gradient_override_map({'Minimum': 'IdentityMaxMinGrad', 'Maximum': 'IdentityMaxMinGrad'}):
    xq = tf.abs(x)
    xq = tf.clip_by_value((xq / scaling), 0, 1)

  xq = round_to_bits(xq, bits=bits, stochastic=stochastic)
  xq = xq * scaling * sign

  return tf.identity(xq, name=name) if name else xq

def quantize_a(x, bits=0, stochastic=False, use_pact=False, trunc_to_bits=0, name=None):

  if bits == 0:
    return x
  if bits == 1:
    return binarize(x, stochastic=stochastic, name=name)

  print('  * quantize_a: bits=%s, use_pact=%r' % (b2s(bits), use_pact))

  axis = list(range(1, x.shape.ndims))
  sign = tf.stop_gradient(tf.sign(x))
  scaling = tf.reduce_max(tf.abs(x), axis=axis, keepdims=True)
  scaling = tf.stop_gradient(scaling)

  g = tf.get_default_graph()
  with g.gradient_override_map({'Minimum': 'IdentityMaxMinGrad', 'Maximum': 'IdentityMaxMinGrad'}):
    xq = tf.abs(x)
    xq = tf.clip_by_value((xq / scaling), 0, 1)

  xq = round_to_bits(xq, bits=bits, stochastic=stochastic, trunc_to_bits=trunc_to_bits)
  xq = xq * scaling * sign

  return tf.identity(xq, name=name) if name else xq


#--------------------------------------------------------------------
# Tests
#--------------------------------------------------------------------

_X1 = [ -0.01,   0.7,  0.25,  0.19,  0.61,  0.59]
_X2 = [ -0.01,  -0.7, -0.25,  0.19,  0.61, -0.59]

_TEST_CASES_W = [
  [_X1, _X1,                                    {'bits': 0}],
  [_X2, _X2,                                    {'bits': 0}],
  [[ -0.01,   0.3,  0.05,  0.21],
   [    -1,     1,     1,     1],               {'bits': 1}],
  [[ -0.01,  -0.3,  0.05, -0.21],
   [    -1,    -1,     1,    -1],               {'bits': 1}],
  [[ -0.01,   0.3,   0.2,  0.14,   0.1,  0.26],
   [   0.0,   0.3,   0.2,   0.1,   0.1,   0.3], {'bits': 2}],
  [[ -0.01,  -0.3,  -0.2,  0.14,  -0.1, -0.26],
   [   0.0,  -0.3,  -0.2,   0.1,  -0.1,  -0.3], {'bits': 2}],
]

_TEST_CASES_A = [
  [_X1, _X1,                                    {'bits': 0, 'post_relu': False}],
  [_X2, _X2,                                    {'bits': 0, 'post_relu': False}],
  [[ -0.01,   0.3,  0.05,  0.21],
   [    -1,     1,     1,     1],               {'bits': 1, 'post_relu': False}],
  [[ -0.01,  -0.3,  0.05, -0.21],
   [    -1,    -1,     1,    -1],               {'bits': 1, 'post_relu': False}],
  [[ -0.01,   0.3,   0.2,  0.16,   0.1,  0.26],
   [   0.0,   0.3,   0.2,   0.2,   0.1,   0.3], {'bits': 2, 'post_relu': False}],
  [[  0.01,   0.3,  0.06,  0.21],
   [   0.0,   0.3,   0.1,   0.2],               {'bits': 2, 'post_relu': True}],
  [[  0.01,   0.7,  0.26,  0.19,  0.61,  0.59],
   [   0.0,   0.7,   0.3,   0.2,   0.6,   0.6], {'bits': 3, 'post_relu': True}],
]

_TEST_CASES_T = [
     # 1-bit mantissa            # 0,    2,     3,      4 bit mantissa
  [[-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875, 0.9375],
   [-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875, 0.9375],    {'bits': None}],
  [[-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875, 0.9375],
   [-0.5, -0.25, 0.125, 0.0625, 0.0,  -0.5,   0.5,    0.5],    {'bits': 0}],
  [[-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875, 0.9375],
   [-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75,  0.75,   0.75],    {'bits': 1}],
  [[-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875, 0.9375],
   [-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875,  0.875],    {'bits': 2}],
  [[-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875, 0.9375],
   [-0.5, -0.25, 0.125, 0.0625, 0.0, -0.75, 0.875, 0.9375],    {'bits': 3}],
]

def run_quantize(func, _x, **kwargs):
  ''' Runs a TF func on _x and returns the results '''
  q = func(tf.Variable(_x), **kwargs)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    return q.eval(session=sess)

def test(test_func, test_cases):
  ''' Runs a test_func through a list of test cases '''
  print('Testing function %s:' % test_func.__name__)
  fails = 0

  for test in test_cases:
    out = run_quantize(test_func, np.array([test[0]]), **test[2])
    ref = np.array([test[1]])
    if not np.allclose(out, ref):
      in_ = np.array([test[0]])
      print(' Failure with args:', test[2])
      print('   in:', in_)
      print('  out:', out)
      print('  ref:', ref)
      fails += 1

  if fails == 0:
    print(' All tests passed!')
  else:
    print(' %d/%d tests failed' % (fails, len(_TEST_CASES_W)))

def test_quantize_w():
  test(quantize_w, _TEST_CASES_W)
def test_quantize_a():
  test(quantize_a, _TEST_CASES_A)
def test_trunc():
  test(trunc, _TEST_CASES_T)

#--------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------
if __name__ == '__main__':
  np.set_printoptions(precision=4)
  test_quantize_w()
  test_quantize_a()
  test_trunc()
