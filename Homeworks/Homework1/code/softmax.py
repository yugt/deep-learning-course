import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        if hidden_dim:
            w1 = np.random.normal(scale=weight_scale,size=(input_dim,hidden_dim))
            b1 = np.zeros(hidden_dim)
            self.params['W1'] = w1
            self.params['b1'] = b1

            w2 = np.random.normal(scale=weight_scale,size=(hidden_dim,num_classes))
            b2 = np.zeros(num_classes)
            self.params['W2'] = w2
            self.params['b2'] = b2
        else:
            w1 = np.random.normal(scale=weight_scale,size=(input_dim,num_classes))
            b1 = np.zeros(num_classes)
            self.params['W1'] = w1
            self.params['b1'] = b1  
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        fc1, cache1 = fc_forward(X, self.params['W1'], self.params['b1'])
        fc2, cache2 = None, None
        if 'W2' in self.params:
            act1, cache_act1 = relu_forward(fc1)
            scores, cache2 = fc_forward(act1, self.params['W2'], self.params['b2'])
        else:
            scores = fc1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dloss = softmax_loss(scores, y)

        l2w1 = np.sum(np.square(self.params['W1']))
        loss += 0.5 * self.reg * l2w1
        if 'W2' in self.params:
            l2w2 = np.sum(np.square(self.params['W2']))
            loss += 0.5 * self.reg * l2w2
            dx2, dw2, db2 = fc_backward(dloss, cache2)
            dw2 += self.reg * self.params['W2']
            grads['W2'] = dw2
            grads['b2'] = db2
            dact1 = relu_backward(dx2, cache_act1)
            dx1, dw1, db1 = fc_backward(dact1, cache1)
            dw1 += self.reg * self.params['W1']
            grads['W1'] = dw1
            grads['b1'] = db1
        else:
            dx1, dw1, db1 = fc_backward(dloss, cache1)
            dw1 += self.reg * self.params['W1']
            grads['W1'] = dw1
            grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

if __name__ == '__main__':
    from torchvision.datasets import MNIST
    import solver

    mnist = MNIST(root="../data/MNIST", train=True, download=True)
    train_idx = int(0.9 * mnist.train_data.shape[0])
    train_data = mnist.train_data.numpy().reshape(mnist.train_data.shape[0], -1)
    train_labels = mnist.train_labels.numpy()

    data = {
        'X_train': train_data[:train_idx],
        'y_train': train_labels[:train_idx],
        'X_val': train_data[train_idx:],
        'y_val': train_labels[train_idx:]
    }
    model = SoftmaxClassifier(hidden_dim=train_data.shape[1], reg=1e-2)
    solver = solver.Solver(model, data, update_rule='adam',
                           optim_config={'learning_rate': 1e-3, },
                           lr_decay=0.9, num_epochs=20, batch_size=1000,
                           print_every=100, verbose=True)
    solver.train()

    mnist = MNIST(root="../data/MNIST", train=False, download=True)
    test_data = mnist.test_data.numpy().reshape(mnist.test_data.shape[0], -1)
    test_labels = mnist.test_labels.numpy()

    score = model.loss(test_data)
    print(np.mean(np.argmax(score, axis=1) == test_labels))