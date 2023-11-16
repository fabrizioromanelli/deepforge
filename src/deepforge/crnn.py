# Class for Convolutional Recurrent Neural Network
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 12/10/2023

from .dnn import DNN
from keras.layers import Input, Dense, concatenate, Conv2D, MaxPooling2D, Flatten, LSTM, Reshape
from keras.models import Model

class CRNN(DNN):
  """
  This class implements the Convolutional Recurrent Neural Network model.

  Attributes
  ----------
  __inputArgs : list
      List of arguments for the input layer.
  __convLayersArgs : list
      List of arguments for the convolutional layers.
  __poolLayersArgs : list
      List of arguments for the pool layers.
  __recLayersArgs : list
      List of arguments for the recurrent layers.
  __outLayersArgs : list
      List of arguments for the output layers.
  __inputLayer : list
      List of input layers.
  __layersIn : list
      List of inner layers.
  __layersOut : list
      List of out layers.
  __modelParams : dict
      Dictionary containing the parameters to be passed to the model for its compilation.
  
  Methods
  -------
  setInputs(inputArgs: list)
      Set the input arguments for the CRNN.
  getInputs()
      Get the input arguments for the CRNN.
  setConvLayers(layersArgs: list)
      Set the convolutional layer arguments for the CRNN.
  getConvLayers()
      Get the convolutional layer arguments for the CRNN.
  setPoolLayers(layersArgs: list)
      Set the pool layer arguments for the CRNN.
  getPoolLayers()
      Get the pool layer arguments for the CRNN.
  setRecurrentLayers(layersArgs: list)
      Set the recurrent layer arguments for the CRNN.
  getRecurrentLayers()
      Get the recurrent layer arguments for the CRNN.
  setOutLayers(outLayersArgs: list)
      Set the output layer arguments for the CRNN.
  getOutLayers()
      Get the output layer arguments for the CRNN.
  setModelConfiguration(modelParams: dict)
      Set the parameters for the CRNN model (used for its compilation).
  getModelConfiguration()
      Get the parameters of the CRNN model (used for its compilation).
  build()
      Build the model.
  """

  def __init__(self, name, inputN=1):
    super().__init__(name,inputN)
    self.__inputArgs: list=[]
    self.__convLayersArgs: list=[]
    self.__poolLayersArgs: list=[]
    self.__recLayersArgs: list=[]
    self.__outLayersArgs: list=[]
    self.__inputLayer: list=[]
    self.__layersIn: list=[]
    self.__layersOut: list=[]
    self.__modelParams: dict=None

  def setInputs(self, inputArgs: list) -> None:
    '''
    This method is used to set the input layers of the CRNN model.

    Parameters
    ----------
    inputArgs : list
      The arguments for the input layers.
    
    Raises
    ------
    AssertionError:
      If the number of inputs is incompatible with the input arguments.

    Returns
    -------
    None
    '''
    assert len(inputArgs) == self.getInputsN(), "[DF] Input arguments are incompatible with the number of inputs."
    self.__inputArgs = inputArgs
    return

  def getInputs(self) -> list:
    '''
    This method is used to get the input layers of the CRNN model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no input arguments set yet.

    Returns
    -------
    list :
      The list of input arguments.
    '''
    if self.__inputArgs is not []:
      return self.__inputArgs
    else:
      raise ValueError('[DF] Model has no input arguments set yet.')

  def setConvLayers(self, layersArgs: list) -> None:
    '''
    This method is used to set the convolutional layers of the CRNN model.

    Parameters
    ----------
    layersArgs : list
      The arguments for the convolutional layers.
    
    Returns
    -------
    None
    '''
    self.__convLayersArgs = layersArgs
    return

  def getConvLayers(self) -> list:
    '''
    This method is used to get the convolutional layers of the CRNN model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no convolutional layers arguments set yet.

    Returns
    -------
    list :
      The list of convolutional layers arguments.
    '''
    if self.__convLayersArgs is not []:
      return self.__convLayersArgs
    else:
      raise ValueError('[DF] Model has no layers arguments set yet.')

  def setPoolLayers(self, layersArgs: list) -> None:
    '''
    This method is used to set the pool layers of the CRNN model.

    Parameters
    ----------
    layersArgs : list
      The arguments for the pool layers.
    
    Returns
    -------
    None
    '''
    self.__poolLayersArgs = layersArgs
    return

  def getPoolLayers(self) -> list:
    '''
    This method is used to get the pool layers of the CRNN model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no pool layers arguments set yet.

    Returns
    -------
    list :
      The list of pool layers arguments.
    '''
    if self.__poolLayersArgs is not []:
      return self.__poolLayersArgs
    else:
      raise ValueError('[DF] Model has no layers arguments set yet.')

  def setRecurrentLayers(self, layersArgs: list) -> None:
    '''
    This method is used to set the recurrent layers of the CRNN model.

    Parameters
    ----------
    layersArgs : list
      The arguments for the recurrent layers.
    
    Returns
    -------
    None
    '''
    self.__recLayersArgs = layersArgs
    return

  def getRecurrentLayers(self) -> list:
    '''
    This method is used to get the recurrent layers of the CRNN model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no recurrent layers arguments set yet.

    Returns
    -------
    list :
      The list of recurrent layers arguments.
    '''
    if self.__recLayersArgs is not []:
      return self.__recLayersArgs
    else:
      raise ValueError('[DF] Model has no layers arguments set yet.')

  def setOutLayers(self, outLayersArgs: list) -> None:
    '''
    This method is used to set the out layers of the CRNN model.

    Parameters
    ----------
    outLayersArgs : list
      The arguments for the out layers.
    
    Returns
    -------
    None
    '''
    self.__outLayersArgs = outLayersArgs
    return

  def getOutLayers(self) -> list:
    '''
    This method is used to get the out layers of the CRNN model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no output layers arguments set yet.

    Returns
    -------
    list :
      The list of out layers arguments.
    '''
    if self.__outLayersArgs is not []:
      return self.__outLayersArgs
    else:
      raise ValueError('[DF] Model has no output layers arguments set yet.')

  def setModelConfiguration(self, **modelParams: dict) -> None:
    '''
    This method is used to set the model configuration for the CRNN model.

    Parameters
    ----------
    modelParams : dict
      The model parameters used in the training/fit phase.
    
    Returns
    -------
    None
    '''
    self.__modelParams = modelParams
    return

  def getModelConfiguration(self) -> dict:
    '''
    This method is used to get the model configuration for the CRNN model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no configuration arguments set yet.

    Returns
    -------
    dict :
      The dictionary of the model parameters.
    '''
    if self.__modelParams is not None:
      return self.__modelParams
    else:
      raise ValueError('[DF] Model has no configuration arguments set yet.')

  def build(self):
    '''
    This method is used to build the model for the CRNN model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    print("[DF] Building model...")
    # Build inputs
    for i in range(0, self.getInputsN()):
      self.__inputLayer.append(Input(**self.__inputArgs[i]))

    # Build inner dense layers
    for j in range(0, self.getInputsN()):
      for k in range(0, len(self.__convLayersArgs[j])):
        if k == 0:
          subnet = Conv2D(**self.__convLayersArgs[j][k])(self.__inputLayer[j])
          subnet = MaxPooling2D(**self.__poolLayersArgs[j][k])(subnet)
        elif k < len(self.__convLayersArgs[j]) - 1:
          subnet = Conv2D(**self.__convLayersArgs[j][k])(subnet)
          subnet = MaxPooling2D(**self.__poolLayersArgs[j][k])(subnet)
        else:
          subnet = Conv2D(**self.__convLayersArgs[j][k])(subnet)
          subnet = Flatten()(subnet)

      subnet = Model(inputs=self.__inputLayer[j], outputs=subnet)

      self.__layersIn.append(subnet.input)
      self.__layersOut.append(subnet.output)

    if self.getInputsN() != 1:
      combined = concatenate(self.__layersOut)
    else:
      combined = self.__layersOut[0]

    # Reshape layer to adapt with LSTM input tensor
    reshaped = Reshape((combined.shape[1],1))(combined)

    for k in range(0, len(self.__recLayersArgs)):
      if k == 0:
        recnet = LSTM(**self.__recLayersArgs[k])(reshaped)
      else:
        recnet = LSTM(**self.__recLayersArgs[k])(recnet)

    # Build output dense layers
    for l in range(0, len(self.__outLayersArgs)):
      if l == 0:
        outnet = Dense(**self.__outLayersArgs[l])(recnet)
      else:
        outnet = Dense(**self.__outLayersArgs[l])(outnet)

    self.setModel(Model(name=self.getName(), inputs=self.__layersIn, outputs=outnet))
    self.getModel().compile(**self.__modelParams)
    print("[DF] Model built!")