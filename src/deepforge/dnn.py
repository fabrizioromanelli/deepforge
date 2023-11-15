# Class for DNN
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 04/10/2023

import os
import tensorflow as tf
import numpy as np
from keras.models import load_model, Model

class DNN:
  """DNN class"""

  # Constructor
  def __init__(self, name: str, inputN: int=1) -> None:
    self.__NAME: str   = name.replace(" ", "")
    self.__inputN: int = inputN
    self.__EPOCHS: int = 0
    self.__model = None

  # Setter/Getter for DNN name
  def setName(self, name: str) -> None:
    self.__NAME = name
    return

  def getName(self) -> str:
    return self.__NAME

  # Setter/Getter for DNN number of inputs
  def setInputsN(self, inputN: int) -> None:
    self.__inputN = inputN
    return

  def getInputsN(self) -> int:
    return self.__inputN

  # Setter/Getter for DNN epochs
  def setEpochs(self, epochs: int) -> None:
    self.__EPOCHS = epochs
    return

  def getEpochs(self) -> int:
    return self.__EPOCHS

  # Setter/Getter for DNN model
  def setModel(self, model: Model) -> None:
    self.__model = model
    return

  def getModel(self) -> Model:
    if self.__model is not None:
      self.__model
      return self.__model
    else:
      print('[DF] Model has not been built yet.')
      return

  # Load the model from a file
  def load(self, filename: str, custom_objects: dict={}, fullpath: bool=False, tflite: bool=False) -> None:
    # TODO when loading a model, update the self.inputN variable
    print("[DF] Loading model...")
    if tflite:
      # TODO implement TFLITE model load
      print("[DF] Load Tflite model not implemented!")
      return

    if fullpath:
      self.__model = load_model(filename)
    else:
      if os.name == 'posix':
        self.__model = load_model('./models/'+filename+'.h5', custom_objects)
      elif os.name == 'nt':
        currentDir = os.getcwd()
        self.__model = load_model(currentDir+'/models/'+filename+'.h5', custom_objects)
    print("[DF] Loaded!")
    return

  # Save the model to a file
  def save(self, filename: str, fullpath: bool=False, tflite: bool=False) -> None:
    if self.__model is None:
      print('[DF] Model has not been built yet.')
      return

    print("[DF] Saving model...")
    if tflite:
      converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
      tflite_model = converter.convert()

    if fullpath:
      if tflite:
        open(filename+'.tflite', 'wb').write(tflite_model)
      else:
        self.__model.save(filename)
    else:
      if os.name == 'posix':
        if tflite:
          open('./models/'+filename+'.tflite', 'wb').write(tflite_model)
        else:
          self.__model.save('./models/'+filename+'.h5') # Python from Linux/WSL2 (saves in ./models)
      elif os.name == 'nt':
        currentDir = os.getcwd()
        if tflite:
          open(currentDir+'/models/'+filename+'.tflite', 'wb').write(tflite_model)
        else:
          self.__model.save(currentDir+'/models/'+filename+'.h5') # Python from Anaconda for Windows (saves in ./models)
    print("[DF] Model saved!")
    return

  # Plot the model summary
  def summary(self) -> None:
    if self.__model is not None:
      self.__model.summary()
      return
    else:
      print('[DF] Model has not been built yet.')
      return

  # Fit model
  def fit(self, **fitParams: dict) -> None:
    if self.__model is not None:
      self.__model.fit(**fitParams)
    else:
      print('[DF] Model has not been built yet.')
      return

  # Make predictions with the model
  @tf.function
  @tf.autograph.experimental.do_not_convert
  def predict(self, x: np.array) -> np.array:
    if self.__model is not None:
      return self.__model(x)
    else:
      print('[DF] Model has not been built yet.')
      return