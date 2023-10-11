# __main__.py

from deepforge import RNN, CNN

def main():
    """Instantiate two DNNs"""

    _RNN = RNN(layers=3,name="Recurrent Neural Network")
    _CNN = CNN(layers=4,name="Convolutional Neural Network")
    print(_RNN.getName())
    print(_CNN.getName())

if __name__ == "__main__":
    main()