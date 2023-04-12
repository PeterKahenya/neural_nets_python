import numpy as np
from nn.activations import SoftMax
from nn.loss import CategoricalCrossEntropyLoss


class SoftmaxCategoricalCrossentropyLoss():

    # def __init__(self):
    #     self.activation = SoftMax()
    #     self.loss = CategoricalCrossEntropyLoss()

    # def forward(self, inputs, y_true):
    #     # Output layer's activation function
    #     self.activation.forward(inputs)
    #     # Set the output
    #     self.outputs = self.activation.outputs
    #     # Calculate and return loss value
    #     return self.loss.calculate(self.outputs, y_true)


    def backward(self,dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
