# deepModel

A python object oriented library to model deep neural networks based on Keras/Tensorflow.

[![License](https://img.shields.io/badge/License-GNU%20GPL-blue)](LICENSE)

## Version

The current version is 0.0.4.

## Overview

deepModel is a Python object-oriented library built on top of Keras and TensorFlow for simplifying the creation and training of deep neural networks. It provides a user-friendly interface for designing, configuring, and training neural network models, making it easier for developers and researchers to work with deep learning.

## Features

- High-level API for defining neural network architectures.
- Modular and extensible design for easy customization.
- Integration with Keras and TensorFlow for efficient training and deployment.
- Support for a variety of neural network types, including feed-forward, convolutional, recurrent, and more.
- Built-in utilities for data preprocessing, evaluation, and visualization.

## Installation

You can install your library using pip:

```bash
pip install deepModel
```

This will install the library with full support for tensorflow-gpu.

## Quick Start

Here's a simple example of how to use the Deep-Model library to create a simple Deep Neural Network via the `multivariateDNN` class:

```python
import numpy as np
# Import the deepModel library
import deepModel as dm

# Initialize the environment
dm.initialize(CPU=20, GPU=1, VERBOSE='2', NPARRAYS=True)

# Make an instance of a multivariate DNN
mDnn = dm.multivariateDNN(name="Simple DNN", inputN=1)

# Set inputs, inner layers and out layers
mDnn.setInputs([{'shape': (2,), 'name': 'Input layer'}])
mDnn.setLayers([[{'units': 16, 'activation': 'elu'}, {'units': 16, 'activation': 'elu'}, {'units': 16, 'activation': 'elu'}, {'units': 3, 'activation': 'linear'}]])
mDnn.setOutLayers([{'units': 1, 'activation': 'linear'}])

# Configure the model
mDnn.setModelConfiguration(optimizer='adam', loss='mse')

# Build the model and print the summary
mDnn.build()
mDnn.summary()

# Train the model
x1 = np.array([2,3,5,6,7], dtype=np.float32)
x2 = np.array([1,2,4,5,6], dtype=np.float32)
X1 = np.array([x1,x2], dtype=np.float32).T
y  = np.array([3,4,6,7,8], dtype=np.float32)

mDnn.fit(x=[X1], y=y, epochs=20, shuffle=True, verbose=0)

# Save the model
mDnn.save('simpleDNN',tflite=False)

# Make a prediction with the model
x1 = np.array([8], dtype=np.float32)
x2 = np.array([7], dtype=np.float32)
X1 = np.array([x1,x2], dtype=np.float32).T
y = mDnn.predict([X1])
print(y.numpy())

# Load the model
mDnnCopy = dm.multivariateDNN(name="simple DNN 2")
mDnnCopy.load("simpleDNN")
```

The output of the previous snippet is:

```
[DF] Building model...
[DF] Model built!
Model: "SimpleDNN"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 Input layer (InputLayer)    [(None, 2)]               0         
                                                                 
 dense_20 (Dense)            (None, 16)                48        
                                                                 
 dense_21 (Dense)            (None, 16)                272       
                                                                 
 dense_22 (Dense)            (None, 16)                272       
                                                                 
 dense_23 (Dense)            (None, 3)                 51        
                                                                 
 dense_24 (Dense)            (None, 1)                 4         
                                                                 
=================================================================
Total params: 647
Trainable params: 647
Non-trainable params: 0
_________________________________________________________________
[DF] Saving model...
[DF] Model saved!
[[1.0306408]]
[DF] Loading model...
[DF] Loaded!
```

The same `multivariateDNN` class can be used to build a more complex DNN model with a custom loss function:

```python
# DNN with 2 input layers and custom loss function example
import numpy as np
from keras.losses import MeanSquaredError
# Import the deepModel library
import deepModel as dm

# Define a custom loss function
def custom_loss(y_true, y_pred):
  mse = MeanSquaredError()
  return mse(y_true, y_pred)

# Make an instance of a multivariate DNN
mDnn2 = dm.multivariateDNN(name="multivariate DNN", inputN=2)

# Set inputs, inner layers and out layers
mDnn2.setInputs([{'shape': (2,), 'name': 'inputLayer1'}, {'shape': (2,), 'name': 'inputLayer2'}])

innerLayers = [[{'units': 32, 'activation': 'elu'}, {'units': 16, 'activation': 'elu'}, {'units': 8, 'activation': 'elu'}, {'units': 3, 'activation': 'linear'}]]
innerLayers.append([{'units': 32, 'activation': 'elu'}, {'units': 16, 'activation': 'elu'}, {'units': 8, 'activation': 'elu'}, {'units': 3, 'activation': 'linear'}])
mDnn2.setLayers(innerLayers)

outputLayers = [{'units': 32, 'activation': 'elu'}, {'units': 1, 'activation': 'linear'}]
mDnn2.setOutLayers(outputLayers)

# Configure the model
mDnn2.setModelConfiguration(optimizer='adam', loss=custom_loss)

# Build the model and print the summary
mDnn2.build()
mDnn2.summary()

# Train the model
x1 = np.array([0.2,0.3,0.5,0.6,0.7], dtype=np.float32)
x2 = np.array([1.1,1.2,1.4,1.5,1.6], dtype=np.float32)
X1 = np.array([x1,x2], dtype=np.float32).T
x3 = np.array([0.2,0.3,0.5,0.6,0.7], dtype=np.float32)
x4 = np.array([1.1,1.2,1.4,1.5,1.6], dtype=np.float32)
X2 = np.array([x3,x4], dtype=np.float32).T
y  = np.array([3.0,4.0,6.0,7.0,8.0], dtype=np.float32)

mDnn2.fit(x=[X1,X2], y=y, epochs=50, shuffle=True, verbose=0)

# Save the model
mDnn2.save('multivariateDNN',tflite=False)

# Make a prediction with the model
x1 = np.array([0.4], dtype=np.float32)
x2 = np.array([1.3], dtype=np.float32)
X1 = np.array([x1,x2], dtype=np.float32).T
x3 = np.array([0.4], dtype=np.float32)
x4 = np.array([1.3], dtype=np.float32)
X2 = np.array([x3,x4], dtype=np.float32).T
y = mDnn2.predict([X1,X2])
print(y.numpy())

# Load the model with the custom loss function
mDnn2Copy = dm.multivariateDNN(name="multivariate DNN 2")
mDnn2Copy.load("multivariateDNN", custom_objects = {'custom_loss': custom_loss})
```

The output of the previous snippet is:

```
[DF] Building model...
[DF] Model built!
Model: "multivariateDNN"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 inputLayer1 (InputLayer)       [(None, 2)]          0           []                               
                                                                                                  
 inputLayer2 (InputLayer)       [(None, 2)]          0           []                               
                                                                                                  
 dense_25 (Dense)               (None, 32)           96          ['inputLayer1[0][0]']            
                                                                                                  
 dense_29 (Dense)               (None, 32)           96          ['inputLayer2[0][0]']            
                                                                                                  
 dense_26 (Dense)               (None, 16)           528         ['dense_25[0][0]']               
                                                                                                  
 dense_30 (Dense)               (None, 16)           528         ['dense_29[0][0]']               
                                                                                                  
 dense_27 (Dense)               (None, 8)            136         ['dense_26[0][0]']               
                                                                                                  
 dense_31 (Dense)               (None, 8)            136         ['dense_30[0][0]']               
                                                                                                  
 dense_28 (Dense)               (None, 3)            27          ['dense_27[0][0]']               
                                                                                                  
 dense_32 (Dense)               (None, 3)            27          ['dense_31[0][0]']               
                                                                                                  
 concatenate_1 (Concatenate)    (None, 6)            0           ['dense_28[0][0]',               
                                                                  'dense_32[0][0]']               
                                                                                                  
 dense_33 (Dense)               (None, 32)           224         ['concatenate_1[0][0]']          
                                                                                                  
 dense_34 (Dense)               (None, 1)            33          ['dense_33[0][0]']               
                                                                                                  
==================================================================================================
Total params: 1,831
Trainable params: 1,831
Non-trainable params: 0
__________________________________________________________________________________________________
[DF] Saving model...
[DF] Model saved!
[[6.0725203]]
[DF] Loading model...
[DF] Loaded!
```

For more detailed usage and examples, please refer to the documentation.

## Documentation

Check out the full documentation for [Keras](https://keras.io/api/) and [Tensorflow](https://www.tensorflow.org/api_docs) for in-depth information on how to use the library.

## License

This project is licensed under the GNU GPL License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, feel free to reach out to [me](mailto:fabrizio.romanelli@gmail.com).
