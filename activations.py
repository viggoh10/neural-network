import numpy as np
import loss

class ReLu:
	def forward(self, inputs, training):
		self.inputs = inputs
		self.output = np.maximum(0, inputs)
	
	def backward(self, dvalues):
		self.dinputs = dvalues.copy()
		self.dinputs[self.inputs <= 0] = 0
		#return np.where(dvalues <= 0, 0, 1)

class Softmax:
	def forward(self, inputs, training):
		self.inputs = inputs
		exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

	def backward(self, dvalues):
		self.dinputs = np.empty_like(dvalues)

		for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			single_output = single_output.reshape(-1,1)
			jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
			self.dinputs[i] = np.dot(jacobian, single_dvalues)

	def predictions(self, outputs):
		return np.argmax(outputs, axis=1)


class Sigmoid():

	def forward(self, inputs, training):
		self.inputs = inputs
		self.output = 1/(1+np.exp(-inputs))
		
	def backward(self, dvalues):
		self.dinputs = dvalues*(1-self.output)*self.output
	
	def predictions(self, outputs):
		return (outputs > 0.5) * 1


class Linear():
	def forward(self, inputs, training):
		self.inputs = inputs
		self.output = inputs

	def backward(self, dvalues):
		self.dinputs = dvalues.copy()
	
		

		
   