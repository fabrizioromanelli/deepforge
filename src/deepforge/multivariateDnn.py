# Class for Multivariate DNN
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 09/10/2023

from .dnn import DNN
from keras.layers import Input, Dense, concatenate
from keras.models import Model

class multivariateDNN(DNN):
  """
  This class implements the multivariate DNN.

  Attributes
  ----------
  __inputArgs : list
      List of arguments for the input layer.
  __layersArgs : list
      List of arguments for the inner dense layers.
  __outLayersArgs : list
      List of arguments for the output layers.
  __inputLayer : list
      List of input layers.
  __layersIn : list
      List of inner dense layers.
  __layersOut : list
      List of out layers.
  __modelParams : dict
      Dictionary containing the parameters to be passed to the model for its compilation.
  
  Methods
  -------
  setInputs(inputArgs: list)
      Set the input arguments for the multivariate DNN.
  getInputs()
      Get the input arguments for the multivariate DNN.
  setLayers(layersArgs: list)
      Set the dense layer arguments for the multivariate DNN.
  getLayers()
      Get the dense layer arguments for the multivariate DNN.
  setOutLayers(outLayersArgs: list)
      Set the output layer arguments for the multivariate DNN.
  getOutLayers()
      Get the output layer arguments for the multivariate DNN.
  setModelConfiguration(modelParams: dict)
      Set the parameters for the multivariate DNN model (used for its compilation).
  getModelConfiguration()
      Get the parameters of the multivariate DNN model (used for its compilation).
  build()
      Build the model.
  """

  def __init__(self, name, inputN=1):
    super().__init__(name,inputN)
    self.__inputArgs: list=[]
    self.__layersArgs: list=[]
    self.__outLayersArgs: list=[]
    self.__inputLayer: list = []
    self.__layersIn: list  = []
    self.__layersOut: list = []
    self.__modelParams: dict=None

  def setInputs(self, inputArgs: list) -> None:
    '''
    This method is used to set the input layers of the multivariate DNN model.

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
    This method is used to get the input layers of the multivariate DNN model.

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

  def setLayers(self, layersArgs: list) -> None:
    '''
    This method is used to set the inner dense layers of the multivariate DNN model.

    Parameters
    ----------
    layersArgs : list
      The arguments for the inner dense layers.
    
    Returns
    -------
    None
    '''
    self.__layersArgs = layersArgs
    return

  def getLayers(self) -> list:
    '''
    This method is used to get the inner dense layers of the multivariate DNN model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no layers arguments set yet.

    Returns
    -------
    list :
      The list of inner layers arguments.
    '''
    if self.__layersArgs is not []:
      return self.__layersArgs
    else:
      raise ValueError('[DF] Model has no layers arguments set yet.')

  def setOutLayers(self, outLayersArgs: list) -> None:
    '''
    This method is used to set the out layers of the multivariate DNN model.

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
    This method is used to get the out layers of the multivariate DNN model.

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
    This method is used to set the model configuration for the multivariate DNN model.

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
    This method is used to get the model configuration for the multivariate DNN model.

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
    This method is used to build the model for the multivariate DNN model.

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
      for k in range(0, len(self.__layersArgs[j])):
        if k == 0:
          subnet = Dense(**self.__layersArgs[j][k])(self.__inputLayer[j])
        else:
          subnet = Dense(**self.__layersArgs[j][k])(subnet)

      subnet = Model(inputs = self.__inputLayer[j], outputs = subnet)

      self.__layersIn.append(subnet.input)
      self.__layersOut.append(subnet.output)

    if self.getInputsN() != 1:
      combined = concatenate(self.__layersOut)
    else:
      combined = self.__layersOut[0]

    # Build output dense layers
    for l in range(0, len(self.__outLayersArgs)):
      if l == 0:
        outnet = Dense(**self.__outLayersArgs[l])(combined)
      else:
        outnet = Dense(**self.__outLayersArgs[l])(outnet)

    self.setModel(Model(name=self.getName(), inputs=self.__layersIn, outputs=outnet))
    self.getModel().compile(**self.__modelParams)
    print("[DF] Model built!")