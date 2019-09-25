from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import json

_DEFAULT_KEY = '_default_'

def read_json_dict(json_name):
  with open(json_name, 'r') as f:
    return json.load(f)

def write_json_dict(json_name, _dict):
  with open(json_name, 'w') as f:
    json.dump(_dict, f, sort_keys=True, indent=2)


def write_or_update_json(json_name, key, val):
  # If file DNE, create a new json file
  if not os.path.isfile(json_name):
    write_json_dict(json_name, {key: val})
  else:
    _dict = read_json_dict(json_name)
    # If val matches the default, do nothing
    if _DEFAULT_KEY in _dict and _dict[_DEFAULT_KEY] == val:
      return
    # Otherwise update the json with new key,value pair
    else:
      _dict[key] = val
      write_json_dict(json_name, _dict)


def access_json(json_name, key):
  if not os.path.isfile(json_name):
    raise ValueError('Json file %s does not exits' % json_name)

  _dict = read_json_dict(json_name)

  if key in _dict:
    return _dict[key]
  elif _DEFAULT_KEY in _dict:
    return _dict[_DEFAULT_KEY]
  else:
    raise ValueError('Key %s not found in %s' % (key, json_name))
