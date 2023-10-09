# Class for Multivariate DNN
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 09/10/2023

from .dnn import DNN
from keras.layers import Input, Dense, concatenate
from keras.models import Model

class multivariateDNN(DNN):
  """Multivariate DNN class"""

  # Constructor
  def __init__(self, layers, name, inputN):
    self.LAYERS = layers
    super().__init__(name,inputN)

  # Setter and getter for layer number
  def setLayerN(self, layers):
    self.LAYERS = layers
    return

  def getLayerN(self):
    if hasattr(self, 'LAYERS'):
      return self.LAYERS
    else:
      print('[DM] Model has no layers set yet.')
      return

  # Setter and getter for Keras Input arguments
  def setInputArgs(self, inputArgs):
    assert len(inputArgs) == self.inputN, "[DM] Input arguments are incompatible with the number of inputs."
    self.inputArgs = inputArgs
    return

  def getInputArgs(self):
    if hasattr(self, 'inputArgs'):
      return self.inputArgs
    else:
      print('[DM] Model has no input arguments set yet.')
      return

  # Setter and getter for Keras Layers arguments
  def setLayersArgs(self, layersArgs):
    assert len(layersArgs) == self.LAYERS, "[DM] Layer arguments are incompatible with the number of layers."
    self.layersArgs = layersArgs
    return

  def getLayersArgs(self):
    if hasattr(self, 'layersArgs'):
      return self.layersArgs
    else:
      print('[DM] Model has no layers arguments set yet.')
      return

  # Build model
  def build(self):
    # Build inputs
    self.inputLayer = []
    for i in range(0, self.inputN):
      self.inputLayer.append(Input(**self.inputArgs[i]))

    # Build inner dense layers
    self.layers = []
    for j in range(0, self.inputN):
      for k in range(0,self.LAYERS):
        if k == 0:
          subnet = Dense(**self.layersArgs[j][k])(self.inputLayer[j])
        elif k < self.LAYERS - 1:
          subnet = Dense(**self.layersArgs[j][k])(subnet)
        else:
          subnet = Model(inputs = self.inputLayer[j], outputs = subnet)

      self.layers.append(subnet.output)

    combined = concatenate(self.layers)

    # TODO
    # z = Dense(32, activation = _activation)(combined)
    # z = Dense(64, activation = _activation)(z)
    # z = Dense(128, activation = _activation)(z)
    # z = Dense(64, activation = _activation)(z)
    # z = Dense(32, activation = _activation)(z)
    # z = Dense(1, activation = "linear")(z)

    # self.model = Model(inputs = [subnet1.input, subnet2.input, subnet3.input], outputs = z)
    # self.model.compile(optimizer = 'adam', loss = 'mse') # = self.mse_pearson_loss)
    # print("Built!")


  # Helpers
  def test(self, **kwargs):
    print(kwargs)
    Input(**kwargs)
    for key,value in kwargs.items():
        print("{}: {}".format(key,value))

  def test2(self, args):
    print(args)
    for arg in args:
      print(arg)
      Input(**arg)