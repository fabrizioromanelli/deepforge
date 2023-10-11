# __main__.py

from deepforge import RNN, CNN

def main():
    """Instantiate two DNNs"""

    _RNN = RNN(name="Recurrent Neural Network")
    _CNN = CNN(name="Convolutional Neural Network")
    print(_RNN.getName())
    print(_CNN.getName())

if __name__ == "__main__":
    main()