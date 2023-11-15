# Class for Recurrent Neural Network
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 11/10/2023

from .dnn import DNN
from keras.layers import Input, Dense, concatenate, LSTM
from keras.models import Model

class RNN(DNN):
  """Recurrent Neural Network class"""

  # Constructor
  def __init__(self, name: str, inputN: int=1):
    super().__init__(name,inputN)
    self.__inputArgs: list=[]
    self.__recLayersArgs: list=[]
    self.__outLayersArgs: list=[]
    self.__modelParams: dict=None
    self.__inputLayer: list=[]
    self.__layersIn: list=[]
    self.__layersOut: list=[]


  # Setter and getter for Keras Input arguments
  def setInputs(self, inputArgs: list) -> None:
    assert len(inputArgs) == self.getInputsN(), "[DF] Input arguments are incompatible with the number of inputs."
    self.__inputArgs = inputArgs
    return

  def getInputs(self) -> list:
    if self.__inputArgs is not []:
      return self.__inputArgs
    else:
      print('[DF] Model has no input arguments set yet.')
      return

  # Setter and getter for recurrent Layer arguments
  def setRecurrentLayers(self, layersArgs: list) -> None:
    self.__recLayersArgs = layersArgs
    return

  def getRecurrentLayers(self) -> list:
    if self.__recLayersArgs is not []:
      return self.__recLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
      return

  # Setter and getter for Keras ouput Layers arguments
  def setOutLayers(self, outLayersArgs: list) -> None:
    self.__outLayersArgs = outLayersArgs
    return

  def getOutLayers(self) -> list:
    if self.__outLayersArgs is not []:
      return self.__outLayersArgs
    else:
      print('[DF] Model has no output layers arguments set yet.')
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
  def build(self):
    print("[DF] Building model...")
    # Build inputs
    for i in range(0, self.getInputsN()):
      self.__inputLayer.append(Input(**self.__inputArgs[i]))

    # Build inner dense layers
    for j in range(0, self.getInputsN()):
      for k in range(0, len(self.__recLayersArgs[j])):
        if k == 0:
          subnet = LSTM(**self.__recLayersArgs[j][k])(self.__inputLayer[j])
        else:
          subnet = LSTM(**self.__recLayersArgs[j][k])(subnet)

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