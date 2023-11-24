# __main__.py

from deepforge import RNN, CNN, CRNN, DAE, multivariateDNN

def main():
    """Instantiate several DNNs"""

    _RNN  = RNN(name="Recurrent Neural Network")
    _CNN  = CNN(name="Convolutional Neural Network")
    _CRNN = CRNN(name="Convolutional Recurrent Neural Network")
    _DAE  = DAE(name="Denoising AutoEncoder")
    _mDNN = multivariateDNN(name="Multivariate Deep Neural Network")
    print(_RNN.getName())
    print(_CNN.getName())
    print(_CRNN.getName())
    print(_DAE.getName())
    print(_mDNN.getName())

if __name__ == "__main__":
    main()