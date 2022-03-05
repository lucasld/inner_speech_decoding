import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def progress_bar(proportion, size=100):
    """Tool to visualize the progress of any process.
    
    :param proportion: proportion of the task that is done
    :type proportion: floating point number
    :param size: target size of the progress-bar, defaults to 100
    :type size: integer, optional
    """
    elements = ['[','', ']', f'{round(proportion*100)}%']
    bar = ''
    for i in range(size):
        p = i / size
        if p <= proportion:
            elements[1] += f'\033[38;2;{int(255*(1-p))};{int(255*p)};{0}m#'
        else:
            elements[1] += ' '
    print('\033[0m'.join(elements), end='\r')


def plot_eeg_data(sample):
    plt.imshow(sample)


class TrainingGrapher:
    def __init__(self, *args, name=None, supxlabel=None, supylabel=None, axs_xlabels=None, axs_ylabels=None, x_scale=None, y_scale=None):
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.plot_shape = args

        plt.ion()
        self.fig, self.axs = plt.subplots(*args)
        # put self.axs in brackets so that its rank becomes 2
        for i in range(2 - (len(self.axs.shape) if type(self.axs) is np.ndarray else 0)):
            self.axs = np.array([self.axs])
        self.lines = [self.axs[y, x].plot([])[0] for y, x in np.ndindex(self.axs.shape)]
        # figure settings
        if name: self.fig.suptitle(name, size='x-large', weight='semibold')
        if supxlabel: self.fig.supxlabel(supxlabel)
        if supylabel: self.fig.supylabel(supylabel)
        # axis settings
        for y, x in np.ndindex(self.axs.shape):
            # set scales
            if x_scale:
                xlim = x_scale if type(x_scale[0]) is int else x_scale[y][x]
                if xlim:
                    self.axs[y, x].set_xlim(xlim)
            if y_scale:
                ylim = y_scale if type(y_scale[0]) is int else y_scale[y][x]
                if ylim: self.axs[y, x].set_ylim(ylim)
            # set labels
            if axs_xlabels:
                self.axs[y, x].set_xlabel(axs_xlabels[y][x])
            if axs_ylabels:
                self.axs[y, x].set_ylabel(axs_ylabels[y, x])
    
    
    def axs_setting(ylabel, xlabel):
        pass
    
    def update(self, *data):
        if len(data) > 1:
            xdata, ydata = data
        else:
            ydata = data[0]
            xdata = [list(range(len(e))) for e in ydata]
        for i, (line, xd, yd) in enumerate(zip(self.lines, xdata, ydata)):
            line.set_data(xd, yd)
            # autoscaling if necesarry
            if not self.x_scale or (self.x_scale[i%self.plot_shape[1]][i//self.plot_shape[1]] is None):
                self.axs[i%self.plot_shape[1], i//self.plot_shape[1]].set_xlim(min(xd), max(xd))
            if not self.y_scale or (self.y_scale[i%self.plot_shape[1]][i//self.plot_shape[1]] is None):
                self.axs[i%self.plot_shape[1], i//self.plot_shape[1]].set_ylim(min(yd) - 0.002, max(yd) + 0.002)
        plt.pause(0.1)
        plt.show()
