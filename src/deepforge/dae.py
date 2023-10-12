# Class for Denoising AutoEncoder
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 11/10/2023

from .dnn import DNN
from keras.layers import Input, Dense, concatenate, BatchNormalization, LeakyReLU
from keras.models import Model

class DAE(DNN):
  """Denoising AutoEncoder class"""

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

  # Setter and getter for encoder Layer arguments
  def setEncoderLayers(self, layersArgs):
    self.encLayersArgs = layersArgs
    return

  def getEncoderLayers(self):
    if hasattr(self, 'encLayersArgs'):
      return self.encLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
      return

  # Setter and getter for hidden Layer arguments
  def setHiddenLayers(self, layersArgs):
    self.hidLayersArgs = layersArgs
    return

  def getHiddenLayers(self):
    if hasattr(self, 'hidLayersArgs'):
      return self.hidLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
      return

  # Setter and getter for decoder Layer arguments
  def setDecoderLayers(self, layersArgs):
    self.decLayersArgs = layersArgs
    return

  def getDecoderLayers(self):
    if hasattr(self, 'decLayersArgs'):
      return self.decLayersArgs
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

    # Build encoder layer(s)
    self.layersIn  = []
    self.layersOut = []
    for inIdx in range(0, self.inputN):
      for encIdx in range(0, self.inputN):
        if encIdx == 0:
          encoderL = Dense(**self.encLayersArgs[inIdx][encIdx])(self.inputLayer[inIdx])
        else:
          encoderL = Dense(**self.encLayersArgs[inIdx][encIdx])(encoderL)
        encoderL = BatchNormalization()(encoderL)
        encoderL = LeakyReLU()(encoderL)

      encoderL = Model(inputs=self.inputLayer[inIdx], outputs=encoderL)

      self.layersIn.append(encoderL.input)
      self.layersOut.append(encoderL.output)

    if self.inputN != 1:
      combined = concatenate(self.layersOut)
    else:
      combined = self.layersOut[0]

    # Build hidden dense layer(s)
    for hidIdx in range(0, len(self.hidLayersArgs)):
      if hidIdx == 0:
        hiddenL = Dense(**self.hidLayersArgs[hidIdx])(combined)
      else:
        hiddenL = Dense(**self.hidLayersArgs[hidIdx])(hiddenL)

    # Build decoder dense layer(s)
    for decIdx in range(0, len(self.decLayersArgs)):
      if decIdx == 0:
        decoderL = Dense(**self.decLayersArgs[decIdx])(hiddenL)
      else:
        decoderL = Dense(**self.decLayersArgs[decIdx])(decoderL)
      decoderL = BatchNormalization()(decoderL)
      decoderL = LeakyReLU()(decoderL)

    # Build output dense layers
    for outIdx in range(0, len(self.outLayersArgs)):
      if outIdx == 0:
        outnet = Dense(**self.outLayersArgs[outIdx])(decoderL)
      else:
        outnet = Dense(**self.outLayersArgs[outIdx])(outnet)

    self.model = Model(name=self.NAME, inputs=self.layersIn, outputs=outnet)
    self.model.compile(**self.modelParams)
    print("[DF] Model built!")