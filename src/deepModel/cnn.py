# Class for CNN
# author: Fabrizio Romanelli
# email : fabrizio.romanelli@gmail.com
# date  : 04/10/2023

from .dnn import DNN

class CNN(DNN):
  """CNN class"""

  # Constructor
  def __init__(self, _layers, _name):
    self.LAYERS = _layers
    super().__init__(_name)

  def setLayers(self, _layers):
    self.LAYERS = _layers
    return

  def getLayers(self):
    return self.LAYERS