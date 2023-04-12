from nn.accuracy import Accuracy_Categorical, Accuracy_Regression
from nn.activations import Linear, ReLU, Sigmoid, SoftMax
from nn.layers import Dropout, Layer
from nn.loss import BinaryCrossentropyLoss, CategoricalCrossEntropyLoss, MeanSquaredErrorLoss
from nn.optimizers import Adam
from nn.model import Model
import nnfs
from nnfs.datasets import sine_data,spiral_data
nnfs.init()


X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()
model.add(Layer(2,512,  weight_regularizer_l2=5e-4,
                        bias_regularizer_l2=5e-4))
model.add(ReLU())
model.add(Dropout(0.1))
model.add(Layer(512,3))
model.add(SoftMax())

model.set(
    loss=CategoricalCrossEntropyLoss(),
    optimizer=Adam(learning_rate=0.05,decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.train(X,y,epochs=10000,print_every=1000,validation_data=(X_test,y_test))







print("\n".join([str(l) for l in model.layers]))






