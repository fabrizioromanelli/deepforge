# Class for DNN
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 04/10/2023

import os
import tensorflow as tf
from keras.models import load_model
# from keras import backend as K

class DNN:
  """DNN class"""

  # Constructor
  def __init__(self, name, inputN=1):
    self.NAME   = name
    self.inputN = inputN

  # Setter/Getter for DNN name
  def setName(self, name):
    self.NAME = name
    return

  def getName(self):
    return self.NAME

  # Setter/Getter for DNN number of inputs
  def setInputsN(self, inputN):
    self.inputN = inputN
    return

  def getInputsN(self):
    return self.inputN

  # Setter/Getter for DNN epochs
  def setEpochs(self, epochs):
    self.EPOCHS = epochs
    return

  def getEpochs(self):
    return self.EPOCHS

  # Getter for DNN model
  def getModel(self):
    if hasattr(self, 'model'):
      self.model
      return self.model
    else:
      print('[DM] Model has not been built yet.')
      return

  # Load the model from a file
  def loadModel(self, filename, fullpath=False, tflite=False):
    if ~hasattr(self, 'model'):
      print('[DM] Model has not been built yet.')
      return

    print("[DM] Loading model...")
    if tflite:
      # TODO implement TFLITE model load
      print("[DM] Load Tflite model not implemented!")
      return

    if fullpath:
      self.model = load_model(filename)
    else:
      if os.name == 'posix':
        self.model = load_model('./model/'+filename+'.h5')
      elif os.name == 'nt':
        self.model = load_model('C:/yourPath/model/'+filename+'.h5')
    print("[DM] Loaded!")
    return

  # Save the model to a file
  def save(self, filename, fullpath=False, tflite=False):
    if ~hasattr(self, 'model'):
      print('[DM] Model has not been built yet.')
      return

    print("[DM] Saving model...")
    if tflite:
      converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
      tflite_model = converter.convert()

    if fullpath:
      if tflite:
        open(filename+'.tflite', 'wb').write(tflite_model)
      else:
        self.model.save(filename)
    else:
      if os.name == 'posix':
        if tflite:
          open('./models/'+filename+'.tflite', 'wb').write(tflite_model)
        else:
          self.model.save('./models/'+filename+'.h5') # Python from Linux/WSL2 (saves in ./models)
      elif os.name == 'nt':
        if tflite:
          open('C:/yourPath/'+filename+'.tflite', 'wb').write(tflite_model)
        else:
          self.model.save('C:/yourPath/'+filename+'.h5') # Python from Anaconda for Windows (saves in C:/yourPath)
    print("[DM] Model saved!")
    return

  # Plot the model summary
  def summary(self):
    if hasattr(self, 'model'):
      self.model.summary()
      return
    else:
      print('[DM] Model has not been built yet.')
      return

  # Fit model
  def fit(self, **fitParams):
    if hasattr(self, 'model'):
      self.model.fit(**fitParams)
    else:
      print('[DM] Model has not been built yet.')
      return

  # Make predictions with the model
  def predict(self):
    if ~hasattr(self, 'model'):
      print('[DM] Model has not been built yet.')
      return

    # TODO implement predict model
    pass