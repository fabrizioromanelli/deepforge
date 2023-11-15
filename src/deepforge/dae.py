# Class for Denoising AutoEncoder
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 13/10/2023

from .dnn import DNN
from keras.layers import Input, Dense
from keras.models import Model

class DAE(DNN):
  """Denoising AutoEncoder class"""

  # Constructor
  def __init__(self, name: str, inputN: int=1) -> None:
    super().__init__(name,inputN)
    self.__encLayersArgs: list = []
    self.__hidLayersArgs: list = []
    self.__decLayersArgs: list = []
    self.__modelParams: dict=None

  # Setter and getter for encoder Layer arguments
  def setEncoderLayer(self, layersArgs: list) -> None:
    self.__encLayersArgs = layersArgs
    return

  def getEncoderLayer(self) -> list:
    if self.__encLayersArgs is not []:
      return self.__encLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
      return

  # Setter and getter for hidden Layer arguments
  def setHiddenLayers(self, layersArgs: list) -> None:
    self.__hidLayersArgs = layersArgs
    return

  def getHiddenLayers(self) -> list:
    if self.__hidLayersArgs is not []:
      return self.__hidLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
      return

  # Setter and getter for decoder Layer arguments
  def setDecoderLayer(self, layersArgs: list) -> None:
    self.__decLayersArgs = layersArgs
    return

  def getDecoderLayer(self) -> list:
    if self.__decLayersArgs is not []:
      return self.__decLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
      return

  # Setter and getter for model configuration
  def setModelConfiguration(self, **modelParams: dict) -> None:
    self.__modelParams = modelParams
    return

  def getModelConfiguration(self) -> dict:
    if self.__modelParams is not None:
      return self.__modelParams
    else:
      print('[DF] Model has no configuration arguments set yet.')
      return

  # Build model
  def build(self) -> None:
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