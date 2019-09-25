from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os, re
import argparse

WEIGHT_NAME_FILTER = '(weights|kernel)'

class LayerParams:
  """ A class which holds all the weights, biases, and other
      params for a single layer.
  """
  def __init__(self, name):
    self.name = name          # name of this layer
    self.param_names = []     # name of this param
    self.param_vars = {}      # TF var of this param
    self.params = {}          # value of this param

  def contains_param(self, name):
    return name in self.param_names

  def add_param(self, name, var, value):
    self.param_names.append(name)
    self.param_vars[name] = var
    self.params[name] = value

  def get_weight(self):
    if 'weights' in self.param_names:
      return self.params['weights']
    elif 'kernel' in self.param_names:
      return self.params['kernel']
    else:
      raise ValueError('LayerParams: cannot figure out weight name')
  
  def get_param(self, name):
    return self.params[name]

  def get_param_var(self, name):
    return self.param_vars[name]

  def assert_consistency(self):
    for name in self.param_names:
      assert(name in self.params)
      assert(name in self.param_vars)

    if 'moving_mean' in self.param_names or 'moving_variance' in self.param_names:
      assert('moving_mean' in self.param_names)
      assert('moving_variance' in self.param_names)

    if 'gamma' in self.param_names or 'beta' in self.param_names:
      assert('gamma' in self.param_names)
      assert('beta' in self.param_names)

  def print(self):
    print("  %s" % self.name)
    for name in self.param_names:
      print('    %-15s -->' % name, self.param_vars[name].shape.as_list())


def find_layer_names(vars):
  """ Given a list of TF variables, returns a list of layer names """
  layer_names = []

  # Search for variables ending with '(weights|kernel):0'
  for v in vars:
    mo = re.search('(.*?)/'+ WEIGHT_NAME_FILTER+':0', v.name)
    if mo and mo.group(1) not in layer_names:
      layer_names.append(mo.group(1))

  return layer_names

def find_layer_params(layer_name, vars):
  """ Given a layer name and a list of vars, construct a LayerParams
      object with all the params for that layer.
  """
  layer = LayerParams(layer_name)

  for v in vars:
    mo = re.search(layer_name+'/(.*?):0', v.name)
    if mo:
      param_name = mo.group(1)
      # weights and biases
      if param_name in ['weights', 'kernel', 'bias']:
        assert(not layer.contains_param(param_name))
        layer.add_param(param_name, v, v.eval())

      # batch norm params
      toks = param_name.split('/')
      if len(toks) == 2 and toks[0] == 'batch_normalization' and \
      toks[1] in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
        assert(not layer.contains_param(toks[1]))
        layer.add_param(toks[1], v, v.eval())
   
  layer.assert_consistency()
  return layer

def find_network_layer_params(vars):
  """ Given a list of vars, finds all weight layers and collects
      params for each layer. Returns a dictionary of layer_names
      to LayerParam objects. """
  network_dict = {}

  # Find all layer names
  print('\nSearching for weight layers...')
  layer_names = find_layer_names(vars)
  for layer_name in layer_names:
    print('  %s' % layer_name)

  # Find params for each layer
  print('\nSearching for layer parameters...')
  for layer_name in layer_names:
    layer = find_layer_params(layer_name, vars)
    network_dict[layer_name] = layer

  return layer_names, network_dict
