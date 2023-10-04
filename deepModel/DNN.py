# Class for DNN
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 04/10/2023

# import tensorflow as tf
# from keras import backend as K

class DNN:
  """DNN class"""

  # Constructor
  def __init__(self, _name):
    # config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 8})
    # sess = tf.compat.v1.Session(config = config)
    # K.set_session(sess)
    self.NAME = _name

  def setName(self, _name):
    self.NAME = _name
    return

  def getName(self):
    return self.NAME