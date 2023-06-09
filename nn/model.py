from nn.activations import SoftMax
from nn.layers import Input, Layer
from nn.loss import CategoricalCrossEntropyLoss
from nn.optimizers import Adam
from nn.utils import SoftmaxCategoricalCrossentropyLoss
import numpy as np
import pickle
import copy


# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.softmax_classifier_output = None

    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    
    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    
    def set_parameters(self,parameters):
        for parameter_set, layer in zip(parameters,self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def save(self,path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs','dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model  
    
    def set(self, *, loss,optimizer,accuracy):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy
    
    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
    
    def finalize(self):
        
        self.input_layer = Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i],"weights"):
                self.trainable_layers.append(self.layers[i])
        if isinstance(self.layers[-1], SoftMax) and isinstance(self.loss, CategoricalCrossEntropyLoss):
            self.softmax_classifier_output = SoftmaxCategoricalCrossentropyLoss()
        self.loss.remember_trainable_layers(self.trainable_layers)
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

    # Train the model
    def train(self,X, y, *, epochs=1, batch_size=None, print_every=1,validation_data=None):
        self.finalize()
        self.accuracy.init(y)
        train_steps = 1
        if batch_size is not None:
            train_steps = int(np.ceil(len(X) / batch_size))
    
        # print(len(X), len(X_val), batch_size, train_steps, validation_steps)
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                outputs = self.forward(batch_X,training=True)
                data_loss, regularization_loss = self.loss.calculate(outputs,batch_y,include_regularization=True)
                batch_loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(outputs)
                accuracy = self.accuracy.calculate(predictions,batch_y)
                self.backward(outputs, batch_y)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {batch_loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

        if validation_data is not None:
            self.evaluate(*validation_data,batch_size=batch_size)
            
    
    def evaluate(self, X_val, y_val,*,batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = int(np.ceil(len(X_val) / batch_size))
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            outputs = self.forward(batch_X, training=False)
            self.loss.calculate(outputs, batch_y)
            predictions = self.output_layer_activation.predictions(outputs)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
                f'acc: {validation_accuracy:.3f}, ' +
                f'loss: {validation_loss:.3f}')


    def forward(self,X,training):
        self.input_layer.forward(X, training)
        for layer in self.layers:

            layer.forward(layer.prev.outputs,training)
            # print(layer.outputs.shape)
        
        return layer.outputs
    
    def backward(self,output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return


        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def predict(self,X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
        batch_output = self.forward(batch_X, training=False)
        output.append(batch_output)
        return np.vstack(output)

        