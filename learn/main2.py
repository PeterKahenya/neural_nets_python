from nn.activations import ReLU, Sigmoid
from nn.layers import Layer
from nn.loss import Loss_BinaryCrossentropy
from nn.optimizers import Adam
import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(samples=1000, classes=2)
y = y.reshape(-1, 1)
# print(X[:2])
# print(y[:999])



dense1 = Layer(2, 64, weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense2 = Layer(64, 1)
# Create Sigmoid activation:
activation2 = Sigmoid()
# Create loss function
loss_function = Loss_BinaryCrossentropy()
# Create optimizer
optimizer = Adam(decay=5e-7)

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.outputs)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function
    # of first layer as inputs
    dense2.forward(activation1.outputs)
    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(dense2.outputs)
    # Calculate the data loss
    data_loss = loss_function.calculate(activation2.outputs, y)

    # Calculate regularization penalty
    regularization_loss = \
    loss_function.regularization_loss(dense1) + \
    loss_function.regularization_loss(dense2)
    # Calculate overall loss
    loss = data_loss + regularization_loss
    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting
    # of True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = (activation2.outputs > 0.5) * 1
    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, '+
        f'loss: {loss:.3f} (' +
        f'data_loss: {data_loss:.3f}, ' +
        f'reg_loss: {regularization_loss:.3f}), ' +
        f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    loss_function.backward(activation2.outputs, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.outputs)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.outputs)
# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.outputs)
# Calculate the data loss
loss = loss_function.calculate(activation2.outputs, y_test)
# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.outputs> 0.5) * 1
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')