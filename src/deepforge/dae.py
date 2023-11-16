# Class for Denoising AutoEncoder
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 13/10/2023

from .dnn import DNN
from keras.layers import Input, Dense
from keras.models import Model

class DAE(DNN):
  """
  This class implements the Denoising Auto-Encoder model.

  Attributes
  ----------
  __encLayersArgs : list
      List of arguments for the encoder layer.
  __hidLayersArgs : list
      List of arguments for the hidden layers.
  __decLayersArgs : list
      List of arguments for the decoder layers.
  __modelParams : dict
      Dictionary containing the parameters to be passed to the model for its compilation.
  
  Methods
  -------
  setEncoderLayer(layersArgs: list)
      Set the encoder layer arguments for the DAE.
  getEncoderLayer()
      Get the encoder layer arguments for the DAE.
  setHiddenLayers(layersArgs: list)
      Set the hidden layer arguments for the DAE.
  getHiddenLayers()
      Get the hidden layer arguments for the DAE.
  setDecoderLayer(layersArgs: list)
      Set the decoder layer arguments for the DAE.
  getDecoderLayer()
      Get the decoder layer arguments for the DAE.
  setModelConfiguration(modelParams: dict)
      Set the parameters for the DAE model (used for its compilation).
  getModelConfiguration()
      Get the parameters of the DAE model (used for its compilation).
  build()
      Build the model.
  """

  def __init__(self, name: str, inputN: int=1) -> None:
    super().__init__(name,inputN)
    self.__encLayersArgs: list = []
    self.__hidLayersArgs: list = []
    self.__decLayersArgs: list = []
    self.__modelParams: dict=None

  def setEncoderLayer(self, layersArgs: list) -> None:
    '''
    This method is used to set the encoder layers of the DAE model.

    Parameters
    ----------
    layersArgs : list
      The arguments for the encoder layers.
    
    Returns
    -------
    None
    '''
    self.__encLayersArgs = layersArgs
    return

  def getEncoderLayer(self) -> list:
    '''
    This method is used to get the encoder layers of the DAE model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no encoder layers arguments set yet.

    Returns
    -------
    list :
      The list of encoder layers arguments.
    '''
    if self.__encLayersArgs is not []:
      return self.__encLayersArgs
    else:
      raise ValueError('[DF] Model has no layers arguments set yet.')

  def setHiddenLayers(self, layersArgs: list) -> None:
    '''
    This method is used to set the hidden layers of the DAE model.

    Parameters
    ----------
    layersArgs : list
      The arguments for the hidden layers.
    
    Returns
    -------
    None
    '''
    self.__hidLayersArgs = layersArgs
    return

  def getHiddenLayers(self) -> list:
    '''
    This method is used to get the hidden layers of the DAE model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no hidden layers arguments set yet.

    Returns
    -------
    list :
      The list of hidden layers arguments.
    '''
    if self.__hidLayersArgs is not []:
      return self.__hidLayersArgs
    else:
      raise ValueError('[DF] Model has no layers arguments set yet.')

  def setDecoderLayer(self, layersArgs: list) -> None:
    '''
    This method is used to set the decoder layers of the DAE model.

    Parameters
    ----------
    layersArgs : list
      The arguments for the decoder layers.
    
    Returns
    -------
    None
    '''
    self.__decLayersArgs = layersArgs
    return

  def getDecoderLayer(self) -> list:
    '''
    This method is used to get the decoder layers of the DAE model.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError:
      If the model has no decoder layers arguments set yet.

    Returns
    -------
    list :
      The list of decoder layers arguments.
    '''
    if self.__decLayersArgs is not []:
      return self.__decLayersArgs
    else:
      raise ValueError('[DF] Model has no layers arguments set yet.')

  def setModelConfiguration(self, **modelParams: dict) -> None:
    '''
    This method is used to set the model configuration for the DAE model.

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
    This method is used to get the model configuration for the DAE model.

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

  def build(self) -> None:
    '''
    This method is used to build the model for the DAE model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    print("[DF] Building model...")

    # Build encoder layer
    encoderLayer = Input(**self.__encLayersArgs)

    # Build hidden dense layer(s)
    for hidIdx in range(0, len(self.__hidLayersArgs)):
      if hidIdx == 0:
        hiddenL = Dense(**self.__hidLayersArgs[hidIdx])(encoderLayer)
      else:
        hiddenL = Dense(**self.__hidLayersArgs[hidIdx])(hiddenL)

    decoderLayer = Dense(**self.__decLayersArgs)(hiddenL)

    self.setModel(Model(name=self.getName(), inputs=encoderLayer, outputs=decoderLayer))
    self.getModel().compile(**self.__modelParams)
    print("[DF] Model built!")