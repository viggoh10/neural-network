import numpy as np

class SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate*(1./(1.+ self.decay*self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum*layer.weight_momentums - self.learning_rate*layer.dW
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum*layer.bias_momentums - self.learning_rate*layer.db
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate*layer.dW
            bias_updates = -self.current_learning_rate*layer.db
        
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


