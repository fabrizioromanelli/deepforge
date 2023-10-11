# Initialization module
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 06/10/2023

import os
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from keras import backend as K

# This function initializes the environment.
# It allows to set the verbosity for the tensorflow module,
# to treat tensors as numpy arrays and to set the configuration
# of the Keras session (both in terms of number of CPUs and GPUs).
def initialize(CPU=1, GPU=1, VERBOSE='0', NPARRAYS=False):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = VERBOSE

  # The following line allows to use tensors as numpy arrays
  # so that the @tf decorator can be used in the class functions.
  if NPARRAYS:
    np_config.enable_numpy_behavior()

  config = tf.compat.v1.ConfigProto(device_count = {'GPU': GPU , 'CPU': CPU})
  sess = tf.compat.v1.Session(config = config)
  K.set_session(sess)