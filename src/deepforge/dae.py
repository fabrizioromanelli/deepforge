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
  def __init__(self, name, inputN=1):
    super().__init__(name,inputN)

  # Setter and getter for encoder Layer arguments
  def setEncoderLayer(self, layersArgs):
    self.encLayersArgs = layersArgs
    return

  def getEncoderLayer(self):
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
  def setDecoderLayer(self, layersArgs):
    self.decLayersArgs = layersArgs
    return

  def getDecoderLayer(self):
    if hasattr(self, 'decLayersArgs'):
      return self.decLayersArgs
    else:
      print('[DF] Model has no layers arguments set yet.')
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

    # Build encoder layer
    encoderLayer = Input(**self.encLayersArgs)

    # Build hidden dense layer(s)
    for hidIdx in range(0, len(self.hidLayersArgs)):
      if hidIdx == 0:
        hiddenL = Dense(**self.hidLayersArgs[hidIdx])(encoderLayer)
      else:
        hiddenL = Dense(**self.hidLayersArgs[hidIdx])(hiddenL)

    decoderLayer = Dense(**self.decLayersArgs)(hiddenL)

    self.model = Model(name=self.NAME, inputs=encoderLayer, outputs=decoderLayer)
    self.model.compile(**self.modelParams)
    print("[DF] Model built!")