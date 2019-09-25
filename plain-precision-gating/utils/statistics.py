from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy

def normal_pdf(x, mean=0., var=1.):
  return np.exp(-0.5*((x-mean)**2)/(2*var)) / np.sqrt(2.*np.pi*var)

def normal_cdf(x):
  return 0.5*(1. + scipy.special.erf(x/np.sqrt(2.)))

def truncated_normal_moments(mean, var, lim=0.):
  """ Returns mean, var of a Gaussian truncated from below """
  std = np.sqrt(var)
  alpha = (lim-mean)/std
  phi = normal_pdf(alpha)
  Phi = normal_cdf(alpha)
  Z = 1. - Phi
  trunc_mean = mean + std*phi/Z
  trunc_var  = var * (1. + alpha*phi/Z - (phi/Z)**2)
  return trunc_mean, trunc_var
