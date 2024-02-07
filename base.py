import numpy as np
import math
from nnfs.datasets import spiral_data, vertical_data
import activations 
from loss import *


class Dense:
	def __init__(self, n_inputs, n_neurons, activation):
		self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1,n_neurons))				
		self.activation = activation

	def forward_pass(self, inputs):
		self.output = self.activation(np.dot(np.array(inputs), np.array(self.weights)) + self.biases)
			

X,y = vertical_data(100,3)
inputs = [[1.0,3.0], [0.1,2.1]]

layer1 = Dense(2, 3, activations.ReLu)
output_layer = Dense(3,3, activations.Softmax)
layer1.forward_pass(X)
output_layer.forward_pass(layer1.output)

softmax_outputs = np.array([[0.2,0.43,0.4], [0.1, 0.01, 0.2], [0.23, 0.24, 0.01]])
class_targets = np.array([[1,0,0], [0,1,0], [0,0,1]])

loss_function = Loss_CCE()
loss = loss_function.calculate(output_layer.output, y)
print(loss)

predictions = np.argmax(output_layer.output, axis=1)
if len(y.shape) == 2:
	y = np.argmax(y, axis=1)
accuracy = np.mean(y==predictions)
print(accuracy)

#print(output_layer.output[:5])






