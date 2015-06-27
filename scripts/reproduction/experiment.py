#!/usr/bin/env python
# -*- coding: utf-8 -*-

__doc__ = """{f}
Usage:
    {f} [--alpha=<alpha>] [--n_epoch=<n_epoch>] [--width1 <v>]  [--width2 <v>] [--feat_map_n_1 <v>] [--feat_map_n_final <v>] [--dropout_rate0 <v>] [--dropout_rate1 <v>] [--dropout_rate2 <v>] [--k_top <v>] [--activation <s>] [--learn <s>] [--skip <v>] [--pretrain <s>] [--datatype <v>] [--shuffle <v>] [--emb_size <v>] [--use_regular <v>] [--regular_c <v>]
Options:
    --n_epoch <n_epoch>      Number of epoch [default: auto].
    --alpha <alpha>          Learning rate [default: 0.001].
    --feat_map_n_1 <v>       Learning rate [default: 3].
    --feat_map_n_final <v>   Learning rate [default: 4].
    --width1 <v>             width [default: 10].
    --width2 <v>             width [default: 7].
    --k_top <v>              k_top [default: 4].
    --dropout_rate0 <v>      dropout rate [default: 0.2].
    --dropout_rate1 <v>      dropout rate [default: 0.4].
    --dropout_rate2 <v>      dropout rate [default: 0.5].
    --activation <s>         activation [default: tanh].
    --learn <s>              activation [default: adam].
    --pretrain <s>           pretrain [default: None].
    --skip <v>               skip_unknown_words [default: 1].
    --shuffle <v>            shuffle data [default: 0].
    --datatype <v>           datatype fine-grained or binary [default: 5].
    --emb_size <v>           embedding size [default: 50].
    --use_regular <v>        use regularizers [default: 0].
    --regular_c <v>          use regular_c [default: 0.0].
    -h --help                Show this screen and exit.
""".format(f=__file__)

'''
    TODO:
        pretrained word embeddings
        visualise word embeddings
        visualise filters
'''

from docopt import docopt
from schema import Schema, And, Or, Use
s = Schema({
    '--n_epoch': Use(int),
    '--alpha': Use(float),
    '--width1': Use(int),
    '--width2': Use(int),
    '--feat_map_n_1': Use(int),
    '--feat_map_n_final': Use(int),
    '--feat_map_n_final': Use(int),
    '--k_top': Use(int),
    '--dropout_rate0': Use(float),
    '--dropout_rate1': Use(float),
    '--dropout_rate2': Use(float),
    '--activation': Use(str),
    '--learn': Use(str),
    '--skip': Use(int),
    '--use_regular': Use(int),
    '--regular_c': Use(float),
    '--shuffle': Use(int),
    '--pretrain': Use(str),
    '--emb_size': Use(int),
    '--datatype': Use(int),
    })
args = docopt(__doc__)
args = s.validate(args)
print args

import os
import sys
import math
import datetime
import time

import theano
import numpy
from SdA import SdA

import theanets

import pems
import util
import plot

def main():
    # setup output directory
    d = datetime.datetime.today()
    output_folder = "out/{}-{}-{}_{}:{}:{}".format(d.year, d.month, d.day, d.hour, d.minute, d.second)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # load dataset
    datasets = load_data(train_days=2, test_days=2, r=2, d=1)

    train_set_x, train_set_y = util.shared_dataset(datasets[0])
    valid_set_x, valid_set_y = util.shared_dataset(datasets[1])
    test_set_x, test_set_y = util.shared_dataset(datasets[2])

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)

    n_input = train_set_x.get_value(borrow=True).shape[1]
    n_output = train_set_y.get_value(borrow=True).shape[1]

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

    predict_fn = sda.build_predict_function()

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

    ###########
    # PREDICT #
    ###########
    y_pred = predict_fn(test_set_x.get_value(borrow=True))
    mae, mre = util.calculate_error_indexes(test_set_y, y_pred)
    print("-*-*RESULT*-*-")
    print("mae={}".format(mae))
    print("mre={}".format(mre))

    # plot
    for i in xrange(n_output):
        filename = "{}.png".format(str(i))
        plot.savefig(filename, test_set_x, y_pred, indexes=[i])

def load_data(r=2, d=1):
    datasets1 = pems.load_data("../../data/PEMS-SF/PEMS_train", from_day=train_from_day, days=train_days, r=r, d=d)
    datasets2 = pems.load_data("../../data/PEMS-SF/PEMS_train", from_day=test_from_day, days=test_days, r=r, d=d)

    dataset_x, dataset_y = util.numpy_dataset(datasets)

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

    return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))

if __name__ == '__main__':
    main()