import numpy as np
import matplotlib.pyplot as plt

import util

outfig = plt.figure()

def figure(Y, Y_pred=None, title='', indexes=None, fig=None):
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

    # calculate width
    widthscale = max(1, n / 8)
    figsize = [min(30, 8 * widthscale), 6]

    # validate indexes
    for i in indexes:
        if i < 0 or Y.shape[1] <= i:
            indexes.remove(i)

    # create a Figure instance if necessary
    fig = plt.figure() if fig is None else fig
    fig.clear()
    fig.set_size_inches(figsize)

    # create axes
    ax = []
    for i,index in enumerate(indexes):
        ax.append(fig.add_subplot(len(indexes),1,i+1))

    # plot the data
    Yt = Y.transpose()
    if Y_pred is None:
        for i,index in enumerate(indexes):
            ax[i].plot(np.array(xrange(n)), Yt[index], "b.-")
    else:
        pred_Yt = Y_pred.transpose()
        for i,index in enumerate(indexes):
            ax[i].plot(np.array(xrange(n)), Yt[index], "b.-")
            ax[i].plot(np.array(xrange(n)), pred_Yt[index], "r.-")

    plt.title(title)

    return fig

def savefig(filename, Y, Y_pred=None, title='', indexes=None):
    '''
    plot the graph
    :param filename: filename
    :param Y: observation data
    :param Y_pred: prediction data
    :param title: title text
    :param from_m: lane to plot
    :param ms:
    :param block:
    :return:
    '''
    fig = figure(Y, Y_pred=Y_pred, title=title, indexes=indexes, fig=outfig)
    plt.savefig(filename)

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
    fig = figure(Y, Y_pred=Y_pred, title=title, indexes=indexes)
    plt.show(block=block)


