# -*- coding: utf-8 -*-
import os
import sys
import time

import theano
import numpy
from SdA import SdA

import theanets

from pems import load_data
import plot

def test_theanets():
    # load dataset
    datasets = load_data("../../data/PEMS-SF/PEMS_train", from_day=0, days=2, r=1, d=1)
    datasets2 = load_data("../../data/PEMS-SF/PEMS_test", from_day=0, days=2, r=1, d=1)

    dataset_x, dataset_y = datasets

    idx = range(0, dataset_x.shape[0])
    numpy.random.shuffle(idx)
    cut = int(0.8 * len(idx))
    train = idx[:cut]
    valid = idx[cut:]

    train_set_x = dataset_x[train]
    train_set_y = dataset_y[train]
    valid_set_x = dataset_x[valid]
    valid_set_y = dataset_y[valid]
    test_set_x, test_set_y = datasets2

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    n_input = dataset_x.shape[1]
    n_output = dataset_y.shape[1]

    def pretrain(exp):
        print('pretraining...')
        train, valid = exp.train(train_set_x, valid_set_x, optimize='pretrain')
        print(' : train[loss]={}, valid[loss]={}'.format(train['loss'], valid['loss']))

    def train(exp, learning_rate=0.1, momentum=0.9):
        print('training...')
        for train, valid in exp.itertrain(train_set, valid_set, learning_rate=learning_rate, momentum=momentum):
            print(' ({},{}): train[loss]={}, valid[loss]={}'.format(learning_rate, momentum, train['loss'], valid['loss']))

    def test(exp):
        pred_y = exp.network.predict(test_set_x)

        # calculate Mean Absolute Percentage Error (MAPE)
        E = test_set_y - pred_y
        absE = numpy.absolute(E)
        mx = numpy.sum(test_set_x) / (test_set_x.shape[0] * test_set_x.shape[1]) # mean of X
        mae = numpy.sum(absE) / (absE.shape[0] * absE.shape[1])
        mre = mae / mx

        # print("Y = \n{}".format(test_set_y))
        # print("Y(pred) = \n{}".format(pred_y))
        # print("E = \n{}".format(E))
        print("MAE = {}".format(mae))
        print("MRE = {}%".format(mre*100.0))

        return pred_y, mae, mre

    def test_and_plot(exp, max_m=n_output, block=False):
        pred_y, mae, mre = test(exp)
        print("MAE = {}".format(mae))
        print("MRE = {}%".format(mre*100.0))
        plot.plot(test_set_y, pred_y, max_m=max_m, block=block)

    # 実験用のネットワークを作る
    exp1 = theanets.Experiment(
        theanets.feedforward.Regressor,
        layers=(
            n_input,
            dict(size=100, activation='linear'),
            dict(size=100, activation='linear'),
            dict(size=100, activation='linear'),
            n_output
        ),
        optimize='sgd',
        activation='linear'
    )

    # train the model
    pretrain(exp1)
    train(exp1)
    test_and_plot(exp1, max_m=2, block=True)


def test_SdA(finetune_lr=0.1, training_epochs=1000,
             pretrain_lr=0.01, pretraining_epochs=15,
             batch_size=1):
    # load dataset
    datasets = load_data("../../data/PEMS-SF/PEMS_train", from_day=0, days=2, r=1, d=1)
    datasets2 = load_data("../../data/PEMS-SF/PEMS_test", from_day=0, days=2, r=1, d=1)

    dataset_x, dataset_y = datasets
    dataset2_x, dataset2_y = datasets2

    idx = range(0, dataset_x.shape[0])
    numpy.random.shuffle(idx)
    cut = int(0.8 * len(idx))
    train = idx[:cut]
    valid = idx[cut:]

    train_set_x = theano.shared(dataset_x[train], borrow=True)
    train_set_y = theano.shared(dataset_y[train], borrow=True)
    valid_set_x = theano.shared(dataset_x[valid], borrow=True)
    valid_set_y = theano.shared(dataset_y[valid], borrow=True)
    test_set_x = theano.shared(dataset2_x, borrow=True)
    test_set_y = theano.shared(dataset2_y, borrow=True)

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)

    n_input = dataset_x.shape[1]
    n_output = dataset_y.shape[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_input,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=n_output
    )

    '''
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))
            print("Pre-training layer {}, epoch {}, cost ".format(i, epoch)),
            print("{}".format(numpy.mean(c)))

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    '''

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=(train_set, valid_set, test_set),
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    # test_theanets()
    test_SdA()