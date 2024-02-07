import numpy as np

def ReLu(inputs):
	return np.maximum(0, inputs)

def Softmax(inputs):
	exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
	return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)


