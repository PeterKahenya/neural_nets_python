import matplotlib.pyplot as plt
import numpy as np
from nn.activations import Linear, ReLU
from nn.layers import Layer
from nn.loss import MeanSquaredErrorLoss
from nn.optimizers import Adam
import nnfs
from nnfs.datasets import sine_data
nnfs.init()
X, y = sine_data()


# Create Dense layer with 1 input feature and 64 output values
dense1 = Layer(1, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLU()
dense2 = Layer(64, 64)
# Create ReLU activation (to be used with Dense layer):
activation2 = ReLU()
# Create second Dense layer with 64 input features (as we take output of previous layer here) and 1 output value
dense3 = Layer(64, 1)
# Create Linear activation:
activation3 = Linear()
# Create loss function
loss_function = MeanSquaredErrorLoss()
# Create optimizer
optimizer = Adam(learning_rate=0.05, decay=1e-3)
# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
accuracy_precision = np.std(y) / 250
# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    dense3.forward(activation2.outputs)
    activation3.forward(dense3.outputs)

    data_loss = loss_function.calculate(activation3.outputs, y)
    # Calculate regularization penalty
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2) + \
        loss_function.regularization_loss(dense3) 
    # Calculate overall loss
    loss = data_loss + regularization_loss
    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.outputs
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f} (' +
        f'data_loss: {data_loss:.3f}, ' +
        f'reg_loss: {regularization_loss:.3f}), ' +
        f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_function.backward(activation3.outputs, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()


X_test, y_test = sine_data()
dense1.forward(X_test)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)
dense3.forward(activation2.outputs)
activation3.forward(dense3.outputs)
plt.plot(X_test, y_test)
plt.plot(X_test, activation3.outputs)
plt.savefig("fifth_try_regression.png")




