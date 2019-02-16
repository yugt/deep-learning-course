import numpy as np

from layers import *


class ConvNet(object):
    """
    A convolutional network with the following architecture:
    
    conv - relu - 2x2 max pool - fc - softmax

    You may also consider adding dropout layer or batch normalization layer. 
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    
    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
        of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        channel = 1
        conv_out_shape = input_dim[1] - filter_size + 1
        w1 = np.random.normal(scale=weight_scale,
            size=(num_filters,channel,filter_size, filter_size))
        self.params['W1'] = w1

        w2 = np.random.normal(scale=weight_scale,
            size=(num_filters * channel * (conv_out_shape//2) * (conv_out_shape//2),
            hidden_dim))
        b2 = np.zeros(hidden_dim)
        self.params['W2'] = w2
        self.params['b2'] = b2

        w3 = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        b3 = np.zeros(num_classes)
        self.params['W3'] = w3
        self.params['b3'] = b3

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
     
 
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv1, cache1 = conv_forward(X, W1)
        pool1, cache_pool1 = max_pool_forward(conv1, pool_param)
        act1, cache_act1 = relu_forward(pool1)

        act1_shape = act1.shape
        batch_size = X.shape[0]
        act1 = np.reshape(act1,
            (act1_shape[0], act1_shape[1] * act1_shape[2] * act1_shape[3]))

        fc2, cache2 = fc_forward(act1, W2, b2)
        act2, cache_act2 = relu_forward(fc2)

        scores, cache3 = fc_forward(act2, W3, b3)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        if y is None:
            return scores
        
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dloss = softmax_loss(scores, y)

        l2w1 = np.sum(np.square(W1))
        loss += 0.5 * self.reg * l2w1

        l2w2 = np.sum(np.square(W2))
        loss += 0.5 * self.reg * l2w2

        l2w3 = np.sum(np.square(W3))
        loss += 0.5 * self.reg * l2w3

        dx3, dw3, db3 = fc_backward(dloss, cache3)
        dw3 += self.reg * W3
        grads['W3'] = dw3
        grads['b3'] = db3

        dact2 = relu_backward(dx3, cache_act2)

        dx2, dw2, db2 = fc_backward(dact2, cache2)
        dw2 += self.reg * W2
        grads['W2'] = dw2
        grads['b2'] = db2

        dx2 = np.reshape(dx2, act1_shape)

        dact1 = relu_backward(dx2, cache_act1)

        dpool1 = max_pool_backward(dx2, cache_pool1)

        dx1, dw1 = conv_backward(dpool1, cache1)
        dw1 += self.reg * W1
        grads['W1'] = dw1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        return loss, grads
  
  
pass



if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import solver

    input_dim = (1, 28, 28)
    mnist = MNIST(root="../data/MNIST", train=True, download=True)
    mnist_train_data = mnist.train_data.numpy()
    mnist_train_labels = mnist.train_labels.numpy()

    train_idx = int(0.9 * mnist_train_data.shape[0])
    train_data = mnist_train_data.reshape(mnist_train_data.shape[0], *input_dim)
    train_labels = mnist_train_labels

    print(train_data.shape)

    data = {
        'X_train': train_data[:train_idx],
        'y_train': train_labels[:train_idx],
        'X_val': train_data[train_idx:],
        'y_val': train_labels[train_idx:]
    }

    mnist = MNIST(root="../data/MNIST", train=False, download=True)
    mnist_test_data = mnist.test_data.numpy()
    mnist_test_labels = mnist.test_labels.numpy()

    test_data = mnist_test_data.reshape(mnist_test_data.shape[0], *input_dim)
    test_labels = mnist_test_labels

    model = ConvNet(input_dim=input_dim)
    cnn_solver = solver.Solver(model, data,
                           update_rule='sgd',
                           optim_config={
                               'learning_rate': 0.05,
                           },
                           num_epochs=1, batch_size=1,
                           print_every=5,
                           verbose=True)
    cnn_solver.train()

    score = model.loss(test_data)
    accuracy = np.mean(np.argmax(score, axis=1) == test_labels)
    print(accuracy)

    model = ConvNet(input_dim=input_dim, reg=1e-2, hidden_dim=20)
    cnn_solver = solver.Solver(model, data,
                           update_rule='sgd',
                           optim_config={
                               'learning_rate': 5e-3,
                           },
                           lr_decay=0.1,
                           num_epochs=5, batch_size=10,
                           print_every=5,
                           verbose=True)
    cnn_solver.train()

    score = model.loss(test_data)
    accuracy = np.mean(np.argmax(score, axis=1) == test_labels)
    print(accuracy)