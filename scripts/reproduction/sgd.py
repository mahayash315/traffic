import os
import sys
import time

import numpy

import theano
import theano.tensor as T

class LinearRegression(object):
    """
    Real-value Regression Class
    """

    def __init__(self, input, n_in, n_out, activation=T.tanh):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type activation: theano.Op or function
        :param activation: theano.tensor activation function

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        lin_output = T.dot(input, self.W) + self.b
        self.y_pred = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            # return mean squared error
            return T.mean(T.sqr(y-self.y_pred))
        else:
            raise NotImplementedError()
