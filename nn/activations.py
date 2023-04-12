import numpy as np


class ReLU:
    def __init__(self):
        self.dinputs = None
        self.outputs = None
        self.inputs = None

    def forward(self, inputs, training):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    def predictions(self,outputs):
        return outputs


class SoftMax:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs, training):
        probabilities = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalized_probs = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        self.outputs = normalized_probs
    
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
    
    def predictions(self,outputs):
        return np.argmax(outputs,axis=1)


# Sigmoid activation
class Sigmoid:
    def forward(self,inputs, training):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.outputs) * self.outputs

    def predictions(self,outputs):
        return (outputs > 0.5) * 1


# Linear activation
class Linear:
    # Forward pass
    def forward(self,inputs, training):
        # Just remember values
        self.inputs = inputs
        self.outputs = inputs
        # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues =
        self.dinputs = dvalues.copy()
    
    def predictions(self,outputs):
        return outputs




