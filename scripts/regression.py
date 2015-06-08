import os
import sys
import time

import numpy
import theanets

import traffic
import plot
import helper

def load_data(dataset, r=0, d=1):
    ''' Loads the dataset and setup experiment

    '''
    # load the dataset
    dataset_x, dataset_y = traffic.load_data(dataset, r=r, d=d)

    # cut the dataset for training, testing, validation
    cut1 = int(0.8 * len(dataset_x)) # 80% for training
    cut2 = int(0.9 * len(dataset_x)) # 10% for validation, 10% for testing
    # cut1 = int(0.9 * len(dataset_x)) # 80% for training
    # cut2 = int(1.0 * len(dataset_x)) # 10% for validation, 10% for testing

    idx = range(0, len(dataset_x))
    numpy.random.shuffle(idx)
    train = idx[:cut1]
    valid = idx[cut1:cut2]
    test = idx[cut2:]

    # idx = range(0, cut2)
    # numpy.random.shuffle(idx)
    # train = idx[:cut1]
    # valid = idx[cut1:]
    # test = range(0, len(dataset_x))

    train_set_x = dataset_x[train]
    train_set_y = dataset_y[train]
    valid_set_x = dataset_x[valid]
    valid_set_y = dataset_y[valid]
    test_set_x = dataset_x[test]
    test_set_y = dataset_y[test]

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def test_regression():
    try:
        print('loading dataset...'),
        datasets = load_data('/Users/masayuki/git/traffic/data/lane.180000.3.xml', r=0, d=1)
        print('done')

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        print("train data: {}".format(len(train_set_x)))
        print("valid data: {}".format(len(valid_set_x)))
        print("test data: {}".format(len(test_set_x)))

        train_set = (train_set_x, train_set_y)
        valid_set = (train_set_x, train_set_y)

        n_input = len(train_set_x[0])
        n_output = len(train_set_y[0])

        # create an experiment
        exp = theanets.Experiment(
                theanets.feedforward.Regressor,
                layers=(
                    n_input,
                    dict(size=100, activation='linear'),
                    n_output
                ),
                optimize='sgd',
                activation='linear'
            )

        def pretrain(num):
            for i in xrange(num):
                train, valid = exp.train(train_set_x, valid_set_x, optimize='pretrain')
                print(' {}: train[loss]={}, valid[loss]={}'.format(i+1, train['loss'], valid['loss']))

        def train(learning_rate=0.01, momentum=0.9, callback=None):
            for train, valid in exp.itertrain(train_set, valid_set, learning_rate=learning_rate, momentum=momentum):
                print(' ({},{}): train[loss]={}, valid[loss]={}'.format(learning_rate, momentum, train['loss'], valid['loss']))
            if callable(callback):
                callback(train=train, valid=valid)

        def test(train=None, valid=None, block=False):
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

            plot.plot(test_set_y, pred_y, block=block)

        # pretrain the model
        print('pretraining...')
        sys.stdout.flush()
        pretrain(1)
        print('done')

        # train the model
        print('training...')
        loss = -1
        # while True:
        #     train, valid = exp.train(train_set, valid_set, learning_rate=0.01, momentum=0.9)
        #     print('  train[loss]={}, valid[loss]={}'.format(train['loss'], valid['loss']))
        #     if (0 <= loss and loss < train['loss']):
        #         break
        #     loss = train['loss']
        train(learning_rate=0.01, momentum=0.9, callback=test)
        train(learning_rate=0.001, momentum=0.99, callback=test)
        train(learning_rate=0.0001, momentum=0.999, callback=test)
        train(learning_rate=0.00001, momentum=0.9999, callback=test)
        train(learning_rate=0.000001, momentum=0.99999, callback=test)
        train(learning_rate=0.0000001, momentum=0.999999, callback=test)
        train(learning_rate=0.00000001, momentum=0.9999999, callback=test)
        train(learning_rate=0.000000001, momentum=0.99999999, callback=test)
        train(learning_rate=0.0000000001, momentum=0.999999999, callback=test)
        train(learning_rate=0.00000000001, momentum=0.9999999999, callback=test)
        print('done')

        # test the model
        test(block=True)
        print("finished.")


    except Exception as e:
        print('error')
        print(str(e))





if __name__ == '__main__':
    test_regression()