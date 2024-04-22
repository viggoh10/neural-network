import numpy as np

class Dense:
	def __init__(self, n_inputs, n_neurons, w_l1=0, b_l1=0, w_l2=0, b_l2=0):
		self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1,n_neurons))				
		self.w_l1=w_l1
		self.b_l1=b_l1
		self.w_l2=w_l2
		self.b_l2=b_l2

	def forward(self, inputs, training):
		self.inputs = inputs
		self.output = np.dot(inputs, self.weights) + self.biases
	
	def backward(self, dvalues):
		self.dW = np.dot(self.inputs.T, dvalues)
		self.db = np.sum(dvalues, axis=0, keepdims=True)

		if self.w_l1 > 0:
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dW += self.w_l1*dL1

		if self.b_l1 > 0:
			dL1 = np.ones_like(self.biases)
			dL1[self.biases < 0] = -1
			self.db += self.b_l1*dL1

		if self.w_l2 > 0:
			self.dW += 2*self.w_l2*self.weights

		if self.b_l2 > 0:
			self.db += 2*self.b_l2*self.biases

		self.dinputs = np.dot(dvalues, self.weights.T)

