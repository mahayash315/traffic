import os
import sys
import time

import numpy
import theanets

import traffic
import plot

def load_data(dataset, r=0, d=1):
    ''' Loads the dataset and setup experiment

    '''
    # load the dataset
    dataset_x, dataset_y = traffic.load_data(dataset, r=r, d=d)

    # cut the dataset for training, testing, validation
    cut1 = int(0.7 * len(dataset_x)) # 80% for training
    cut2 = int(0.8 * len(dataset_y)) # 10% for validation, 10% for testing

    idx = range(len(dataset_x))
    numpy.random.shuffle(idx)

    train = idx[:cut1]
    train_set_x = dataset_x[train]
    train_set_y = dataset_y[train]
    valid = idx[cut1:cut2]
    valid_set_x = dataset_x[valid]
    valid_set_y = dataset_y[valid]
    test = idx[cut2:]
    test_set_x = dataset_x[test]
    test_set_y = dataset_y[test]

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def test_regression():
    try:
        print('loading dataset...'),
        datasets = load_data('/Users/masayuki/traffic/data/lane.180000.2.xml', r=0, d=1)
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
                layers=(n_input,100,100,n_output),
                optimize='sgd',
                activation='tanh'
            )

        def pretrain(num):
            for i in xrange(num):
                train, valid = exp.train(train_set_x, valid_set_x, optimize='pretrain')
                print(' {}: train[loss]={}, valid[loss]={}'.format(i+1, train['loss'], valid['loss']))

        def train(num, base_lr=0.01, gamma=0.1, stepsize=1, momentum=0.9, callback=None):
            learning_rate = base_lr
            for i in xrange(num):
                #train, valid = exp.train(train_set, valid_set, learning_rate=learning_rate, momentum=momentum)
                for train, valid in exp.itertrain(train_set, valid_set, learning_rate=learning_rate, momentum=momentum):
                    print(' {}({},{}): train[loss]={}, valid[loss]={}'.format(i+1, learning_rate, momentum, train['loss'], valid['loss']))
                if callable(callback):
                    callback(i, train=train, valid=valid)
                if i % stepsize == 0:
                    learning_rate *= gamma

        def test(n, train=None, valid=None, block=False):
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

            plot.plot(test_set_y, pred_y, title="{}".format(n), block=block)

        # pretrain the model
        print('pretraining...')
        sys.stdout.flush()
        pretrain(1)
        print('done')

        # train the model
        print('training...')
        #train(3, callback=test)
        train(1, base_lr=0.01, momentum=0.9, callback=test)
        train(1, base_lr=0.001, momentum=0.99, callback=test)
        train(1, base_lr=0.0001, momentum=0.999, callback=test)
        print('done')

        # test the model
        test(10, block=True)
        print("finished.")


    except Exception as e:
        print('error')
        print(str(e))





if __name__ == '__main__':
    test_regression()