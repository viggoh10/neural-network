import numpy as np

inputs = [1,3,4]
weights = [1,2,3]


class Layer: 
	def __init__(self, weights, inputs, activation):
		self.weights = weights 
		self.inputs = inputs
		self.output = []				
		self.activation = activation

	def forward_pass(self):
		self.output = np.dot(self.inputs, self.weights)
			

def ReLu(inputs):
	return np.maximum(0, inputs)


layer = Layer(weights, inputs, ReLu)
layer.forward_pass()
print(layer.output)






