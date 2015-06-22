# -*- coding: utf-8 -*-
__author__ = 'masayuki'

import sys
import numpy
import theanets
import traffic
import plot

# def load_data(self, filename, r, d):
#     ''' Loads the dataset and setup experiment
#
#     '''
#     # load the dataset
#     dataset_x, dataset_y = traffic.load_data(filename, r=r, d=d)
#
#     # cut the dataset for training, testing, validation
#     cut1 = int(0.8 * len(dataset_x)) # 80% for training
#     cut2 = int(0.9 * len(dataset_x)) # 10% for validation, 10% for testing
#     idx = range(0, len(dataset_x))
#     numpy.random.shuffle(idx)
#     train = idx[:cut1]
#     valid = idx[cut1:cut2]
#     test = range(0, len(dataset_x))
#
#     train_set_x = dataset_x[train]
#     train_set_y = dataset_y[train]
#     valid_set_x = dataset_x[valid]
#     valid_set_y = dataset_y[valid]
#     test_set_x = dataset_x[test]
#     test_set_y = dataset_y[test]
#
#     if (1 <= self.debug_level):
#         print("train data: {}".format(len(train_set_x)))
#         print("valid data: {}".format(len(valid_set_x)))
#         print("test data: {}".format(len(test_set_x)))
#
#     self.n_input = len(train_set_x[0])
#     self.n_output = len(train_set_y[0])
#
#     rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
#     return rval

# 標準化のために割る値, FIXME: 本当は各レーンの最大 Traffic Volume で割って標準化したデータを使うべき
DATA_DIVIDE_VALUE = 100

class Experiment:
    def __init__(self, r=0, d=1, debug_level=0):
        self.debug_level = debug_level
        self.r = r
        self.d = d
        self.n_input = 0
        self.n_output = 0
        self.traindata = None
        self.validdata = None
        self.testdata = None

    def setDebug(self, debug_level):
        self.debug_level = debug_level

    def setTrainData(self, filename):
        # load the dataset
        dataset_x, dataset_y = traffic.load_data(filename, r=self.r, d=self.d)

        # NORMALIZE
        dataset_x = numpy.divide(dataset_x, DATA_DIVIDE_VALUE)
        dataset_y = numpy.divide(dataset_y, DATA_DIVIDE_VALUE)

        # cut the dataset
        cut = int(0.8 * len(dataset_x)) # 80% for training, 20% for validating
        idx = range(0, len(dataset_x))
        numpy.random.shuffle(idx)
        train = idx[:cut]
        valid = idx[cut:]

        # set the dataset
        self.traindata = (dataset_x[train], dataset_y[train])
        self.validdata = (dataset_x[valid], dataset_y[valid])
        self.n_input = len(dataset_x[0])
        self.n_output = len(dataset_y[0])
        if (1 <= self.debug_level):
            print("train data: {}".format(len(self.traindata[0])))
            print("valid data: {}".format(len(self.validdata[0])))

    def setTestData(self, filename):
        # load the dataset
        dataset_x, dataset_y = traffic.load_data(filename, r=self.r, d=self.d)

        # NORMALIZE
        dataset_x = numpy.divide(dataset_x, DATA_DIVIDE_VALUE)
        dataset_y = numpy.divide(dataset_y, DATA_DIVIDE_VALUE)

        # set the dataset
        self.testdata = (dataset_x, dataset_y)

    def get_n_input(self):
        '''
        :return: ネットワークの入力層に必要なユニット数
        '''
        return self.n_input

    def get_n_outupt(self):
        '''
        :return: ネットワークの出力層に必要なユニット数
        '''
        return self.n_output

    def pretrain(self, exp):
        '''
        与えられたネットワーク exp を事前学習する
        :param exp:
        :return:
        '''
        train_set_x, train_set_y = self.traindata
        valid_set_x, valid_set_y = self.validdata

        def pretrain():
            train, valid = exp.train(train_set_x, valid_set_x, optimize='pretrain')
            if (2 <= self.debug_level):
                print(' : train[loss]={}, valid[loss]={}'.format(train['loss'], valid['loss']))

        # pretrain the model
        if (1 <= self.debug_level):
            print('pretraining...')
            sys.stdout.flush()
        pretrain()
        if (1 <= self.debug_level):
            print('done')

    def train(self, exp, n=10):
        '''
        与えたネットワーク exp で学習を行う
        :param exp: ネットワーク
        :param n: 学習回数
        :return:
        '''

        train_set_x, train_set_y = self.traindata
        valid_set_x, valid_set_y = self.validdata

        def train(learning_rate=0.01, momentum=0.9, callback=None):
            for train, valid in exp.itertrain(self.traindata, self.validdata, learning_rate=learning_rate, momentum=momentum):
                if (2 <= self.debug_level):
                    print(' ({},{}): train[loss]={}, valid[loss]={}'.format(learning_rate, momentum, train['loss'], valid['loss']))
            if callable(callback):
                callback(train=train, valid=valid)

        # train the model
        if (1 <= self.debug_level):
            print('training...')
        learning_rate = 0.1
        momentum = 0.9
        for i in xrange(n):
            train(learning_rate, momentum, None)
            learning_rate *= 0.1
            #momentum += 9.0/pow(10,i+2)
            #momentum = min(momentum, 0.999999999999)
        if (1 <= self.debug_level):
            print('done')

    def test(self, exp, block=False):
        '''
        与えたネットワーク exp でテストを行う
        :param exp: ネットワーク
        :return:
        '''

        test_set_x, test_set_y = self.testdata

        def test():
            # predict
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
            if (1 <= self.debug_level):
                print("MAE = {}".format(mae))
                print("MRE = {}%".format(mre*100.0))

            return pred_y, mae, mre

        def test_and_plot(train=None, valid=None, block=False):
            pred_y, mae, mre = test()
            print("MAE = {}".format(mae))
            print("MRE = {}%".format(mre*100.0))
            plot.plot(test_set_y, pred_y, block=block)

        # test the model
        test_and_plot(block=block)
        if (1 <= self.debug_level):
            print("finished.")

def test_networks():
    # experiment をつくる
    bed = Experiment(r=2, d=1, debug_level=2)

    # データ準備
    bed.setTrainData("../data/cross3ltl_full_3/lane.129600.1.xml")
    bed.setTestData("../data/cross3ltl_full_3/lane.129600.2.xml")

    # ネットワーク作成準備
    n_input = bed.get_n_input()
    n_output = bed.get_n_outupt()

    # 実験用のネットワークを作る
    exp1 = theanets.Experiment(
        theanets.feedforward.Regressor,
        layers=(
            n_input,
            dict(size=100, activation='linear'),
            dict(size=100, activation='linear'),
            n_output
        ),
        optimize='sgd',
        activation='linear'
    )

    # 事前学習
    bed.pretrain(exp1)

    # 学習と評価
    bed.train(exp1, 1)
    bed.test(exp1, False)

    bed.train(exp1, 1)
    bed.test(exp1, False)

    bed.train(exp1, 1)
    bed.test(exp1, True)

if __name__ == '__main__':
    try:
        test_networks()
    except Exception as e:
        print('error')
        print(str(e))
