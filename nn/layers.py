import numpy as np


class Layer:
    def __init__(self, input_dim, n_neurons,weight_regularizer_l1=0, weight_regularizer_l2=0,bias_regularizer_l1=0, bias_regularizer_l2=0):
        """
            Initialize matrix of size input_dim(rows) by n_neurons(columns) so that ith column contains weights leading to output neuron i
        """
        self.outputs = None
        self.inputs = None
        self.weights = 0.01 * np.random.randn(input_dim, n_neurons)

        self.biases = np.zeros((1,n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self,weights,biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs, training):
        """
            The ith row of self.output matrix are the outputs when shown example i
        """
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, dvalues):

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims =True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases


        self.dinputs = np.dot(dvalues, self.weights.T) # propagated error


# Dropout
class Dropout:
    # Init
    def __init__(self, rate):
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.outputs = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate,size=inputs.shape) / self.rate
        self.outputs = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Input:
    def forward(self, inputs, training):
        self.outputs = inputs



