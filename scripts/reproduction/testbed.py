# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime

import theano
import numpy
from SdA import SdA

import theanets
import cPickle

import pems
import util
import plot

class TestBed:
    def __init__(self, sda, output_folder=None):
        self.sda = sda
        self._init_output_(output_folder)
        self.working_folder = os.getcwd()
        self.outstate_filename = "state.save"

    @classmethod
    def new(cls, n_ins, hidden_layers_sizes, n_outs, output_folder=None):
        numpy_rng = numpy.random.RandomState(89677)
        sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=n_outs
        )

        return cls(sda, output_folder)

    @classmethod
    def load(cls, state_file, output_folder=None):
        if not os.path.isfile(state_file):
            raise Exception("file not found: {}".format(state_file))

        f = file(state_file, 'rb')
        sda = cPickle.load(f)
        f.close()

        return cls(sda, output_folder)

    def _init_output_(self, output_folder):
        if output_folder is None:
            d = datetime.datetime.today()
            output_folder = "out/{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}".format(d.year, d.month, d.day, d.hour, d.minute, d.second)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
        self.output_folder = output_folder

    def _before_output_(self):
        os.chdir(self.output_folder)

    def _after_output_(self):
        os.chdir(self.working_folder)

    def save_state(self):
        self._before_output_()

        f = file(self.outstate_filename, 'wb')
        cPickle.dump(self.sda, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

        self._after_output_()

    def pretrain(self, train_set_x, epochs=15, learning_rate=0.1, batch_size=1):
        # get functions
        pretraining_fns = self.sda.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size

        print('... pre-training the model')
        start_time = time.clock()
        ## Pre-train layer-wise
        corruption_levels = [.1, .2, .3]
        for i in xrange(self.sda.n_layers):
            # go through pretraining epochs
            for epoch in xrange(epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index, corruption=corruption_levels[i], lr=learning_rate))
                print("Pre-training layer {}, epoch {}, cost {}".format(i, epoch, numpy.mean(c)))
            self.save_state()

        end_time = time.clock()
        print('The pretraining code for file {} ran for {}'.format(os.path.split(__file__)[1], (end_time - start_time) / 60.))

    def finetune(self, train_set, valid_set, test_set, epochs=15, learning_rate=0.1, batch_size=1):
        # get functions
        train_fn, validate_model, test_model = self.sda.build_finetune_functions(
            datasets=(train_set, valid_set, test_set),
            batch_size=batch_size,
        )

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set[0].get_value(borrow=True).shape[0]
        n_train_batches /= batch_size

        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
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
        best_iter = -1

        print('... finetunning the model, learning_rate={}'.format(learning_rate))
        while (epoch < epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index, lr=learning_rate)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch {}, minibatch {}/{}, validation error {}'.format(epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

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
                        print('     epoch {}, minibatch {}/{}, test error of best model {}s'.format(epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

            self.save_state()

        end_time = time.clock()
        print('Optimization complete with best validation score of {} on iteration {}, with test performance {}'.format(best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print('The training code for file {} ran for {}'.format(os.path.split(__file__)[1], ((end_time - start_time) / 60.)))

    def predict(self, x):
        predict_fn = self.sda.build_predict_function()
        return predict_fn(util.get_ndarray(x))

def test_SdA(state_file=None, output_folder=None):
    # load data
    datasets = load_data(r=2, d=1)

    train_set_x, train_set_y = util.shared_dataset(datasets[0])
    valid_set_x, valid_set_y = util.shared_dataset(datasets[1])
    test_set_x, test_set_y = util.shared_dataset(datasets[2])

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)

    n_input = train_set_x.get_value(borrow=True).shape[1]
    n_output = train_set_y.get_value(borrow=True).shape[1]

    # prepare output folder
    if output_folder is None:
        d = datetime.datetime.today()
        output_folder = "out/{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}".format(d.year, d.month, d.day, d.hour, d.minute, d.second)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

    # instantiate TestBed
    if state_file is None:
        bed = TestBed.new(n_input, [400, 400, 400], n_output, output_folder)
    else:
        bed = TestBed.load(state_file)


    ######################
    # PRETRAIN THE MODEL #
    ######################
    bed.pretrain(test_set_x, epochs=1, learning_rate=0.1, batch_size=1)

    ########################
    # FINETUNING THE MODEL #
    ########################
    bed.finetune(train_set, valid_set, test_set, epochs=1000, learning_rate=0.1, batch_size=1)
    bed.finetune(train_set, valid_set, test_set, epochs=1000, learning_rate=0.01, batch_size=1)
    bed.finetune(train_set, valid_set, test_set, epochs=1000, learning_rate=0.001, batch_size=1)
    bed.finetune(train_set, valid_set, test_set, epochs=1000, learning_rate=0.0001, batch_size=1)
    bed.finetune(train_set, valid_set, test_set, epochs=1000, learning_rate=0.00001, batch_size=1)

    ###########
    # PREDICT #
    ###########
    y_pred = bed.predict(test_set_x)

    mae, mre, rmse = util.calculate_error_indexes(test_set_y, y_pred)
    print("-*-*RESULT*-*-")
    print("mae={}".format(mae))
    print("mre={}".format(mre))
    print("rmse={}".format(rmse))

    # plot
    os.chdir(output_folder)
    cut = min(10*144, test_set_x.get_value(borrow=True).shape[0])
    plot_y = test_set_x.get_value(borrow=True)[:cut]
    plot_y_pred = y_pred[:cut]
    for i in xrange(n_output):
        filename = "{}.png".format(str(i))
        plot.savefig(filename, plot_y, plot_y_pred, indexes=[i])

def load_data(r=2, d=1):
    datasets = pems.load_data("../../data/PEMS-SF/PEMS_sorted", from_day=0, days=90, r=r, d=d)

    dataset_x, dataset_y = util.numpy_dataset(datasets)

    idx = range(0, dataset_x.shape[0])
    cut1= int(0.6 * len(idx))
    trainvalid = idx[:cut1]
    test = idx[cut1:]

    numpy.random.shuffle(trainvalid)
    cut2 = int(0.8 * len(trainvalid))
    train = trainvalid[:cut2]
    valid = trainvalid[cut2:]

    train_set_x = dataset_x[train]
    train_set_y = dataset_y[train]
    valid_set_x = dataset_x[valid]
    valid_set_y = dataset_y[valid]
    test_set_x = dataset_x[test]
    test_set_y = dataset_y[test]

    return ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))


if __name__ == '__main__':
    # test_theanets()
    test_SdA()