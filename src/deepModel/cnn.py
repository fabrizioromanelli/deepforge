# Class for Convolutional Neural Network
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 04/10/2023

from .dnn import DNN

class CNN(DNN):
  """CNN class"""

  # Constructor
  def __init__(self, layers, name):
    self.LAYERS = layers
    super().__init__(name)

  def setLayers(self, layers):
    self.LAYERS = layers
    return

  def getLayers(self):
    return self.LAYERS