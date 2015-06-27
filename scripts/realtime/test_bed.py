import time

import numpy
import theano
from theano import tensor as T

from dnn.SdA import SdA
from generator import SimpleGenerator
from visualizer import Visualizer

class TestBed:
    def __init__(self, window_size=1, m=1, r=2, batch_size=1):
        self.r = r
        self.window_size = 1
        self.n_batches = (self.window_size / batch_size)
        self.n_input = m*(r+1)
        self.n_output = m
        self.data = [[0 for j in xrange(m)] for i in xrange(window_size + (r+1))]
        self.x_value = numpy.zeros((window_size, self.n_input), dtype=theano.config.floatX)
        self.x = theano.shared(self.x_value, borrow=True)
        self.y_value = numpy.zeros((window_size, self.n_output), dtype=theano.config.floatX)
        self.y = theano.shared(self.y_value, borrow=True)

        numpy_rng = numpy.random.RandomState(89677)

        print '... building the model'
        # construct the stacked denoising autoencoder class
        self.sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=self.n_input,
            hidden_layers_sizes=[10],
            n_outs=self.n_output
        )

        # retrieving functions
        self.pretraining_fns = self.sda.pretraining_functions(
            train_set_x=self.x,
            batch_size=batch_size
        )
        self.train_fn = self.sda.build_finetune_function(
            train_set_x=self.x,
            train_set_y=self.y,
            batch_size=batch_size,
        )
        self.predict_fn = self.sda.build_prediction_function()

    def supply(self, y):
        self.data.append(y)
        while self.window_size + (self.r+1) < len(self.data):
            self.data.pop(0)

        for i in xrange(self.x_value.shape[0]-1):
            self.x_value[i] = self.x_value[i+1]
            self.y_value[i] = self.y_value[i+1]
        self.x_value[-1] = [self.data[-1-i%(self.r+1)][int(i/(self.r+1))] for i in xrange(self.x_value.shape[1])]
        self.y_value[-1] = y

    def pretrain(self, pretraining_epochs, pretraining_lr):
        corruption_levels = [.1, .2, .3]
        for i in xrange(self.sda.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(self.n_batches):
                    c.append(self.pretraining_fns[i](index=batch_index, corruption=corruption_levels[i], lr=pretraining_lr))
                print('Pre-training layer {}, epoch {}, cost '.format(i, epoch)),
                print("{}".format(numpy.mean(c)))

    def finetune(self, finetunning_epochs, finetunning_lr=0.1):
        done_looping = False
        epoch = 0
        while (epoch < finetunning_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_batches):
                minibatch_avg_cost = self.train_fn(minibatch_index, lr=finetunning_lr)
                print(" epoch {}: cost={}".format(epoch, minibatch_avg_cost))

    def predict(self):
        return self.predict_fn(self.x_value)


def main(m=1, r=3):
    gen = SimpleGenerator(num=m)
    bed = TestBed(m=m, r=r)
    vis = Visualizer()

    # pretrain
    for i in xrange(10):
        bed.supply(gen.next())
    bed.pretrain(10, pretraining_lr=0.1)

    for i,y in enumerate(gen):
        y_pred = bed.predict()
        print("{}: y={}, y_pred={}".format(i, y, y_pred))
        vis.append(y, y_pred)
        bed.supply(y)
        bed.finetune(10, finetunning_lr=0.1)
        time.sleep(1.0)

if __name__ == '__main__':
    main()