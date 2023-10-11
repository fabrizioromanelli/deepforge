# Class for Recurrent Neural Network
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 11/10/2023

from .dnn import DNN
from keras.layers import Input, Dense, concatenate, CuDNNLSTM
from keras.models import Model

class RNN(DNN):
  """Recurrent Neural Network class"""

  # Constructor
  def __init__(self, name, inputN=1):
    super().__init__(name,inputN)

  # Setter and getter for Keras Input arguments
  def setInputs(self, inputArgs):
    assert len(inputArgs) == self.inputN, "[DF] Input arguments are incompatible with the number of inputs."
    self.inputArgs = inputArgs
    return

  def getInputs(self):
    if hasattr(self, 'inputArgs'):
      return self.inputArgs
    else:
      print('[DF] Model has no input arguments set yet.')
      return

  # Setter and getter for recurrent Layer arguments
  def setRecurrentLayers(self, layersArgs):
    self.recLayersArgs = layersArgs
    return

  def getRecurrentLayers(self):
    if hasattr(self, 'recLayersArgs'):
      return self.recLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
      return

  # Setter and getter for Keras ouput Layers arguments
  def setOutLayers(self, outLayersArgs):
    self.outLayersArgs = outLayersArgs
    return

  def getOutLayers(self):
    if hasattr(self, 'outLayersArgs'):
      return self.outLayersArgs
    else:
      print('[DF] Model has no output layers arguments set yet.')
      return

  # Setter and getter for model configuration
  def setModelConfiguration(self, **modelParams):
    self.modelParams = modelParams
    return

  def getModelConfiguration(self):
    if hasattr(self, 'modelParams'):
      return self.modelParams
    else:
      print('[DF] Model has no configuration arguments set yet.')
      return

  # Build model
  def build(self):
    print("[DF] Building model...")
    # Build inputs
    self.inputLayer = []
    for i in range(0, self.inputN):
      self.inputLayer.append(Input(**self.inputArgs[i]))

    # Build inner dense layers
    self.layersIn  = []
    self.layersOut = []
    for j in range(0, self.inputN):
      for k in range(0, len(self.recLayersArgs[j])):
        if k == 0:
          subnet = CuDNNLSTM(**self.recLayersArgs[j][k])(self.inputLayer[j])
        else:
          subnet = CuDNNLSTM(**self.recLayersArgs[j][k])(subnet)

      subnet = Model(inputs = self.inputLayer[j], outputs = subnet)

      self.layersIn.append(subnet.input)
      self.layersOut.append(subnet.output)

    if self.inputN != 1:
      combined = concatenate(self.layersOut)
    else:
      combined = self.layersOut[0]

    # Build output dense layers
    outLayersN = len(self.outLayersArgs)

    for l in range(0, outLayersN):
      if l == 0:
        outnet = Dense(**self.outLayersArgs[l])(combined)
      else:
        outnet = Dense(**self.outLayersArgs[l])(outnet)

    self.model = Model(name=self.NAME, inputs=self.layersIn, outputs=outnet)
    self.model.compile(**self.modelParams)
    print("[DF] Model built!")