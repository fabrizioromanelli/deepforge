# Initialization module
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 06/10/2023

import os
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

def initialize(CPU: int=1, GPU: int=1, VERBOSE: str='0', NPARRAYS: bool=False):
  '''
  This function initializes the environment.
  It allows to set the verbosity for the tensorflow module,
  to treat tensors as numpy arrays and to set the configuration
  of the Keras session (both in terms of number of CPUs and GPUs).

  Parameters
  ----------
  CPU : int
        Number of CPUs to be used in the computations.
  GPU : int
        Number of GPUs to be used in the computations.
  VERBOSE : str
        Log level ('0' means no logs).
  NPARRAYS : bool
        If True, allows to use tensors as numpy arrays.
  
  Returns
  -------
  None
  '''

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = VERBOSE

  # The following line allows to use tensors as numpy arrays
  # so that the @tf decorator can be used in the class functions.
  if NPARRAYS:
    np_config.enable_numpy_behavior()

  config = tf.compat.v1.ConfigProto(device_count = {'GPU': GPU , 'CPU': CPU})
  sess = tf.compat.v1.Session(config = config)
  tf.compat.v1.keras.backend.set_session(sess)