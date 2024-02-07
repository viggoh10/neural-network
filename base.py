import numpy as np

class Dense:
	def __init__(self, n_inputs, n_neurons, activation):
		self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1,n_neurons))				
		self.activation = activation

	def forward_pass(self, inputs):
		self.output = np.dot(np.array(inputs), np.array(self.weights)) + self.biases
			

def ReLu(inputs):
	return np.maximum(0, inputs)

"""def initalizer(prev_layer_size, layer_size):
	return {
		'W': np.zeros((prev_layer_size,layer_size)),
		'b': np.zeros((1, layer_size))
	}"""

inputs = [[1.0,3.0], [0.1,2.1]]

layer = Dense(2, 3, ReLu)
layer.forward_pass(inputs)

print(layer.output[:5])






