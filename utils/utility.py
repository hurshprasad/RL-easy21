from conf.constant import CONST
import numpy as np
import _pickle as cPickle
import os
import pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def save_data(filename, data):
    fp = open(filename + '.pickle', 'wb')
    cPickle.dump(data, fp)
    fp.close()


def load_data(filename):
    assert os.path.isfile(filename + '.pickle'), "file doesn't exist"
    fp = open(filename + '.pickle', 'rb')
    data = cPickle.load(fp)
    fp.close()
    return data


def write_dictionary_tofile(filename, data_dictionary):
    fp = open(filename, 'w')

    for entry in data_dictionary:
        value = data_dictionary[entry]
        fp.write(entry.replace(':', '\t') + '\t' + str(value) + "\n")

    fp.close()


def init_dict(real = False):

    dictionary = dict()
    for dealer in range(0, 11):
        for player in range(0, CONST.EASY21 + 1):
            for action in [CONST.HIT, CONST.STICK]:
                dictionary["%d:%d:%d" % (dealer, player, action)] = 0.0 if real else 0

    return dictionary


def plot_value_function(model, title, filename):
    x = range(1, 11)
    y = range(1, 21)
    X, Y = np.meshgrid(x, y)
    Z = model.get_value_function(x, y)

    # Start from 12 on Y-axis
    X = X[11:]
    Y = Y[11:]
    Z = Z[11:]

    fig = pylab.figure()
    ax = Axes3D(fig)
    pylab.title(title)
    ax.set_xlabel("Dealer Showing")
    pylab.xticks(range(1, 11))
    ax.set_ylabel("Player Sum")
    pylab.yticks(range(12, 21))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0,
                    antialiased=False)
    pylab.savefig(filename)


def plot_learning_curve(mse):
    # plot the mse vs lambda's
    pylab.figure()
    figManager = pylab.get_current_fig_manager()
    pylab.plot(CONST.LAMBDAS, mse[:, -1])
    pylab.xlabel('lambda')
    pylab.ylabel('MSE')
    pylab.title('Mean Squared Error vs Lambda')
    pylab.savefig(CONST.OUTPUT_PATH + 'mse_trace')
    #pylab.show()

    # plot the mse vs episode number for lambda = 0 and 1
    pylab.figure()
    figManager = pylab.get_current_fig_manager()
    pylab.plot(range(CONST.EPISODES), mse[0,:],
               range(CONST.EPISODES), mse[-1,:])
    pylab.xlabel('Episode')
    pylab.ylabel('MSE')
    pylab.title('Mean Squared Error vs Episode Number')
    pylab.legend(['Lambda=0', 'Lambda=1'])
    pylab.savefig(CONST.OUTPUT_PATH + 'mse_episode')
    #pylab.show()
