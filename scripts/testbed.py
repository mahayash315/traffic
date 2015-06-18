# -*- coding: utf-8 -*-
__author__ = 'masayuki'

import sys
import numpy
import theanets
import traffic
import plot

class TestBed:
    def __init__(self, filename, r=0, d=1, debug_level=0):
        self.debug_level = debug_level
        self.filename = filename
        self.r = r
        self.d = d
        self.datasets = self.load_data(filename, r, d)

    def setDebug(self, debug_level):
        self.debug_level = debug_level

    def load_data(self, filename, r, d):
        ''' Loads the dataset and setup experiment

        '''
        # load the dataset
        dataset_x, dataset_y = traffic.load_data(filename, r=r, d=d)

        # cut the dataset for training, testing, validation
        cut1 = int(0.5 * len(dataset_x)) # 80% for training
        cut2 = int(0.6 * len(dataset_x)) # 10% for validation, 10% for testing
        idx = range(0, len(dataset_x))
        numpy.random.shuffle(idx)
        train = idx[:cut1]
        valid = idx[cut1:cut2]
        test = range(0, len(dataset_x))

        train_set_x = dataset_x[train]
        train_set_y = dataset_y[train]
        valid_set_x = dataset_x[valid]
        valid_set_y = dataset_y[valid]
        test_set_x = dataset_x[test]
        test_set_y = dataset_y[test]

        if (1 <= self.debug_level):
            print("train data: {}".format(len(train_set_x)))
            print("valid data: {}".format(len(valid_set_x)))
            print("test data: {}".format(len(test_set_x)))

        self.n_input = len(train_set_x[0])
        self.n_output = len(train_set_y[0])

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
        return rval

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

    def test(self, exp, n=10, do_pretrain=True, block=False):
        '''
        与えたネットワーク exp で学習, テストを行う
        :param exp: ネットワーク
        :param n: 学習回数
        :return:
        '''
        try:
            train_set_x, train_set_y = self.datasets[0]
            valid_set_x, valid_set_y = self.datasets[1]
            test_set_x, test_set_y = self.datasets[2]

            train_set = (train_set_x, train_set_y)
            valid_set = (train_set_x, train_set_y)

            def pretrain():
                train, valid = exp.train(train_set_x, valid_set_x, optimize='pretrain')
                if (2 <= self.debug_level):
                    print(' : train[loss]={}, valid[loss]={}'.format(train['loss'], valid['loss']))

            def train(learning_rate=0.01, momentum=0.9, callback=None):
                for train, valid in exp.itertrain(train_set, valid_set, learning_rate=learning_rate, momentum=momentum):
                    if (2 <= self.debug_level):
                        print(' ({},{}): train[loss]={}, valid[loss]={}'.format(learning_rate, momentum, train['loss'], valid['loss']))
                if callable(callback):
                    callback(train=train, valid=valid)

            def test(train=None, valid=None):
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

                return pred_y

            def test_and_plot(train=None, valid=None, block=False):
                pred_y = test(train=train, valid=valid)
                plot.plot(test_set_y, pred_y, block=block)


            # pretrain the model
            if do_pretrain:
                if (1 <= self.debug_level):
                    print('pretraining...')
                    sys.stdout.flush()
                pretrain()
                if (1 <= self.debug_level):
                    print('done')

            # train the model
            if (1 <= self.debug_level):
                print('training...')
            learning_rate = 0.1
            momentum = 0.9
            for i in xrange(n):
                train(learning_rate, momentum, test)
                learning_rate *= 0.1
                #momentum += 9.0/pow(10,i+2)
                #momentum = min(momentum, 0.999999999999)
            if (1 <= self.debug_level):
                print('done')

            # test the model
            test_and_plot(block=block)
            if (1 <= self.debug_level):
                print("finished.")

        except Exception as e:
            print('error')
            print(str(e))


def test_networks():
    # testbed をつくる
    # bed = TestBed("../data/lane.180000.3.xml", r=2, d=1)
    bed = TestBed("../data/cross3ltl_full_3/lane.129600.xml", r=2, d=1, debug_level=2)
    n_input = bed.get_n_input()
    n_output = bed.get_n_outupt()

    # 実験用のネットワークを作る
    exp1 = theanets.Experiment(
        theanets.feedforward.Regressor,
        layers=(
            n_input,
            dict(size=100, activation='linear'),
            n_output
        ),
        optimize='sgd',
        activation='linear'
    )
    exp2 = theanets.Experiment(
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
    exp3 = theanets.Experiment(
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
    # exp4 = theanets.Experiment(
    #     theanets.feedforward.Regressor,
    #     layers=(
    #         n_input,
    #         dict(size=400, activation='linear'),
    #         dict(size=400, activation='linear'),
    #         dict(size=400, activation='linear'),
    #         n_output
    #     ),
    #     optimize='sgd',
    #     activation='linear'
    # )

    # 実験する
    bed.test(exp1, 20)
    bed.test(exp2, 20)
    bed.test(exp3, 20, block=True)
    # bed.test(exp4, 20, block=True)

    # ブロック
    # print raw_input("何かキーを押すと終了")



if __name__ == '__main__':
    test_networks()
