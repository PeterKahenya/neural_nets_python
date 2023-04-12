import numpy as np




class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            # L1 regularization - weights. Calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            # L1 regularization - biases. Calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        
        return regularization_loss

    # Calculates the data and regularization losses. Given model output and ground truth values
    def calculate(self, output, y,*,include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += data_loss
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return loss
        return data_loss,self.regularization_loss()
    
    # Calculates accumulated loss
    def calculate_accumulated(self,*, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
   

    
    
class CategoricalCrossEntropyLoss(Loss):
    def forward(self,y_pred,y_true):
        samples_count = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        # print(y_true.shape)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples_count),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Losses
        losses = -np.log(correct_confidences)
        # print(losses)
        return losses

    def backward(self,dvalues,y_true):
        samples_count = len(dvalues)
        labels_count = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels_count)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples_count


# Binary cross-entropy loss
class BinaryCrossentropyLoss(Loss):
    # Forward pass
    def forward(self,y_pred, y_true):
        # Clip data to prevent division by 0 Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +(1 - y_true) * np.log(1 - y_pred_clipped))
        # print(sample_losses)
        sample_losses = np.mean(sample_losses, axis=-1)
        # print(sample_losses)
        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self,dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -(1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        

class MeanSquaredErrorLoss(Loss):

    def forward(self,y_pred,y_true):
        losses = np.mean((y_pred-y_true)**2, axis=-1)
        return losses
    
    def backward(self,dvalues,y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Absolute Error loss
class MeanAbsoluteErrorLoss(Loss):
    # L1 loss
    def forward(self,y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

