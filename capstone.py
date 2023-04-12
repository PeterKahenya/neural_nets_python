import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import numpy as np
from nn.accuracy import Accuracy_Categorical
from nn.activations import ReLU, SoftMax
from nn.layers import Layer
from nn.loss import CategoricalCrossEntropyLoss
from nn.model import Model
from nn.optimizers import Adam
import nnfs

nnfs.init()
np.set_printoptions(linewidth=200)

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file),cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]
    return X, y, X_test, y_test


X, y, X_test, y_test = create_data_mnist(FOLDER)

EPOCHS=5
BATCH_SIZE=128

# model = Model()
# model.add(Layer(X.shape[1], 128))
# model.add(ReLU())
# model.add(Layer(128, 128))
# model.add(ReLU())
# model.add(Layer(128,10))
# model.add(SoftMax())

# model.set(
#     loss = CategoricalCrossEntropyLoss(),
#     optimizer=Adam(decay=1e-3),
#     accuracy=Accuracy_Categorical()
# )

# model.train(X,y,validation_data=(X_test,y_test),epochs=EPOCHS,batch_size=BATCH_SIZE,print_every=100)

# model.save_parameters('fashion_mnist.params')

# model.save('fashion_mnist.model')


# Load the model
model = Model.load('fashion_mnist.model')
# Evaluate the model
# model.evaluate(X_test, y_test)
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)
for prediction in predictions:
    print(fashion_mnist_labels[prediction])


