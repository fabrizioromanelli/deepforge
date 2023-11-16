# Class for DNN
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 04/10/2023

import os
import tensorflow as tf
import numpy as np
from keras.models import load_model, Model

class DNN:
  """
  This class implements the super class DNN
  for a generic Deep Neural Network model.

  Attributes
  ----------
  NAME : str
         Name of the instance model.
  inputN : int
         Number of inputs for the DNN.
  EPOCHS : int
         Number of epochs to be used for the training phase.
  model : Model
         Model (in Keras form).
  
  Methods
  -------
  setName(name: str)
      Set the DNN instance name.
  getName()
      Get the DNN instance name.
  setInputsN(inputN: int)
      Set the number of inputs for the DNN.
  getInputsN()
      Get the number of inputs for the DNN.
  setEpochs(epochs: int)
      Set the number of epochs to be used in the training/fit phase.
  getEpochs()
      Get the number of epochs to be used in the training/fit phase.
  setModel(model: Model)
      Set the Keras model in the class attribute.
  getModel()
      Get the Keras model from the class attribute.
  load(filename: str, custom_objects: dict, fullpath: bool, tflite: bool)
      Load the Keras model from a file.
  save(filename: str, fullpath: bool, tflite: bool)
      Save the Keras model to a file.
  summary()
      Get the summary of the model.
  fit(**fitParams: dict)
      Trains/fits the model.
  predict(x: np.array)
      Make a prediction with a pre-trained model.
  """

  # Constructor
  def __init__(self, name: str, inputN: int=1) -> None:
    self.__NAME: str   = name.replace(" ", "")
    self.__inputN: int = inputN
    self.__EPOCHS: int = 0
    self.__model = None

  def setName(self, name: str) -> None:
    '''
    This method is used to set the DNN instance name.

    Parameters
    ----------
    name : str
          The DNN instance name.
    
    Returns
    -------
    None
    '''
    self.__NAME = name
    return

  def getName(self) -> str:
    '''
    This method is used to get the DNN instance name.

    Parameters
    ----------
    None
    
    Returns
    -------
    str :
      The DNN instance name.
    '''
    return self.__NAME

  def setInputsN(self, inputN: int) -> None:
    '''
    This method is used to set the DNN inputs.

    Parameters
    ----------
    inputN : int
          The number of inputs to the DNN.
    
    Returns
    -------
    None
    '''
    self.__inputN = inputN
    return

  def getInputsN(self) -> int:
    '''
    This method is used to get the DNN inputs number.

    Parameters
    ----------
    None
    
    Returns
    -------
    int :
      The DNN input numbers.
    '''
    return self.__inputN

  def setEpochs(self, epochs: int) -> None:
    '''
    This method is used to set the number of epochs for the DNN training.

    Parameters
    ----------
    epochs : int
          The number of epochs for the training of the DNN.
    
    Returns
    -------
    None
    '''
    self.__EPOCHS = epochs
    return

  def getEpochs(self) -> int:
    '''
    This method is used to get the number of epochs for the DNN training.

    Parameters
    ----------
    None
    
    Returns
    -------
    int :
      The number of epochs for the training of the DNN.
    '''
    return self.__EPOCHS

  def setModel(self, model: Model) -> None:
    '''
    This method is used to set the DNN Keras model.

    Parameters
    ----------
    model : Model
          The Keras model of the DNN.
    
    Returns
    -------
    None
    '''
    self.__model = model
    return

  def getModel(self) -> Model:
    '''
    This method is used to get the DNN Keras model.

    Parameters
    ----------
    None
    
    Raises
    ------
    ValueError
      If the model has not been built yet.

    Returns
    -------
    Model :
      The Keras model of the DNN.
    '''
    if self.__model is not None:
      self.__model
      return self.__model
    else:
      raise ValueError('[DF] Model has not been built yet.')

  def load(self, filename: str, custom_objects: dict={}, fullpath: bool=False, tflite: bool=False) -> None:
    '''
    This method is used to load the DNN Keras model from a file.

    Parameters
    ----------
    filename : str
      The filename of the Keras model.
    custom_objects : dict
      A dictionary containing the custom objects that has been used in the Keras model.
    fullpath : bool
      If true, the user can give the full path in the filename and load the model from there.
    tflite : bool
      If true, deepforge will load a tflite model, rather than a Keras standard model.
    
    Raises
    ------
    NotImplementedError
      If the component is not yet implemented.

    Returns
    -------
    None
    '''
    # TODO when loading a model, update the self.inputN variable
    print("[DF] Loading model...")
    if tflite:
      # TODO implement TFLITE model load
      raise NotImplementedError('[DF] Load Tflite model not implemented!')

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

  def save(self, filename: str, fullpath: bool=False, tflite: bool=False) -> None:
    '''
    This method is used to save the DNN Keras model to a file.

    Parameters
    ----------
    filename : str
      The filename of the Keras model.
    fullpath : bool
      If true, the user can give the full path in the filename and save the model to that path.
    tflite : bool
      If true, deepforge will save a tflite model, rather than a Keras standard model.

    Returns
    -------
    None
    '''
    if self.__model is None:
      raise ValueError('[DF] Model has not been built yet.')

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

  def summary(self) -> None:
    '''
    This method is used to print a summary of the DNN Keras model.

    Parameters
    ----------
    None
    
    Raises
    ------
    ValueError
      If the model has not been built yet.

    Returns
    -------
    None
    '''
    if self.__model is not None:
      self.__model.summary()
      return
    else:
      raise ValueError('[DF] Model has not been built yet.')

  def fit(self, **fitParams: dict) -> None:
    '''
    This method is used to fit the DNN Keras model, given the fit parameters.

    Parameters
    ----------
    fitParams : dict
      The dictionary containing the parameters to be used in the fitting phase
      of the model.

    Raises
    ------
    ValueError
      If the model has not been built yet.

    Returns
    -------
    None
    '''
    if self.__model is not None:
      self.__model.fit(**fitParams)
    else:
      raise ValueError('[DF] Model has not been built yet.')

  @tf.function
  @tf.autograph.experimental.do_not_convert
  def predict(self, x: np.array) -> np.array:
    '''
    This method uses the DNN Keras model to make predictions, given the input.

    Parameters
    ----------
    x : np.array
      The Numpy array containing the input(s).

    Raises
    ------
    ValueError
      If the model has not been built yet.

    Returns
    -------
    np.array :
      The predicted value(s).
    '''
    if self.__model is not None:
      return self.__model(x)
    else:
      raise ValueError('[DF] Model has not been built yet.')