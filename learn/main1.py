from nn.activations import ReLU, SoftMax
from nn.layers import Dropout, Layer
from nn.loss import CategoricalCrossEntropyLoss
from nn.model import Model
import numpy as np
from nn.optimizers import SGD, AdaGrad, Adam, RMSProp
from nn.utils import SoftmaxCategoricalCrossentropyLoss
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X, y = spiral_data(samples=1000, classes=3)

# define architecture
dense1 = Layer(2,64,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
# print(dense1.weights[:1])
activation1 = ReLU()
dropout1 = Dropout(0.1)
dense2 = Layer(64,3)
# activation2 = SoftMax()
# loss_layer = CategoricalCrossEntropyLoss()
activation_loss = SoftmaxCategoricalCrossentropyLoss()

## optimizer (learn)
# optimizer = SGD(learning_rate=1.,decay=1e-7,momentum=0.6)
# optimizer = AdaGrad(decay=1e-4)
# optimizer = RMSProp(decay=1e-4)
# optimizer = RMSProp(learning_rate=0.02, decay=1e-5,rho=0.999)
optimizer = Adam(learning_rate=0.05, decay=5e-5)


for epoch in range(10001):
    # forward
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dropout1.forward(activation1.outputs)
    dense2.forward(dropout1.outputs)
    # activation2.forward(dense2.outputs)
    data_loss = activation_loss.forward(dense2.outputs,y)

    # Calculate regularization penalty
    regularization_loss = activation_loss.loss.regularization_loss(dense1) + activation_loss.loss.regularization_loss(dense2)
    # print(regularization_loss)
    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets calculate values along 1st axis
    predictions = np.argmax(activation_loss.outputs, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f} (' +
                f'data_loss: {data_loss:.3f}, ' +
                f'reg_loss: {regularization_loss:.3f}), ' +
                f'lr: {optimizer.current_learning_rate}')

    activation_loss.backward(activation_loss.outputs, y)
    # activation2.backward(loss_layer.dinputs)
    dense2.backward(activation_loss.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


# Create test dataset and run tests
X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
loss = activation_loss.forward(dense2.outputs, y_test)
predictions = np.argmax(activation_loss.outputs, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')





