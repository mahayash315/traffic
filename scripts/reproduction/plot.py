import numpy as np
import matplotlib.pyplot as plt

import util

def plot(Y, Y_pred=None, title='', indexes=None, block=True):
    '''
    plot the graph
    :param Y: observation data
    :param Y_pred: prediction data
    :param title: title text
    :param from_m: lane to plot
    :param ms:
    :param block:
    :return:
    '''
    Y = util.get_ndarray(Y)
    Y_pred = None if Y_pred is None else util.get_ndarray(Y_pred)

    # count the data
    n = Y.shape[0]
    indexes = range(0,Y.shape[1]) if indexes is None else indexes

    # validate indexes
    for i in indexes:
        if i < 0 or Y.shape[1] <= i:
            indexes.remove(i)

    # create a Figure instance
    fig = plt.figure()

    # create axes
    ax = []
    for i in indexes:
        ax.append(fig.add_subplot(len(indexes),1,i+1))

    # plot the data
    Yt = Y.transpose()
    if Y_pred == None:
        for i in indexes:
            ax[i].plot(np.array(xrange(n)), Yt[i], "b.-")
    else:
        pred_Yt = Y_pred.transpose()
        for i in indexes:
            ax[i].plot(np.array(xrange(n)), Yt[i], "b.-")
            ax[i].plot(np.array(xrange(n)), pred_Yt[i], "r.-")

    plt.title(title)
    plt.show(block=block)


