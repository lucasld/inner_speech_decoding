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


def plot_inter_train_results(results, figure_title, pretrain_res=None, key='val_accuracy'):
    """Plot training progress of pretraining models.
    
    :param results: list of subjects inter training results
    :type results: list of lists of dicts
    :param figure_title: saving location for created plot
    :type figure_title: string
    :param pretrain_res: results of the pretraining that should be pretended to
        the visualization
    :type pretrain_res: list of list of dicts
    :param key: key of result type to be visualized
    :type key: string
    """
    # initialize subject accumulators
    pretrain_sub_acc = []
    train_sub_acc = []
    # intialize subplots
    fig, axs = plt.subplots(len(results), figsize=(7.5, 15))
    # Set the y ticks and limit
    plt.setp(axs, yticks=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], ylim=[0, 0.5])
    # enumerate through all subjects contained in the results
    for i, subject_results in enumerate(results):
        # accumulate all training historys
        history_acc = []
        ax = axs[i] if len(results) > 1 else axs
        ax.set_title(f'Subject {i+1}')
        # iterate through all n independend "k-fold calls"
        for n in range(len(subject_results)):
            # pretrain data for one k-fold call
            pretrain_data = pretrain_res[i][n][key] if pretrain_res else []
            if len(pretrain_data):
                ax.plot(pretrain_data, alpha=0.08)
            # iterate through the k folds
            for k_fold in subject_results[n]:
                # combine pretrain history with after-pretrain history
                comp_data = pretrain_data + k_fold[key]
                # plot line
                ax.plot(range(len(pretrain_data)-1, len(comp_data)), [pretrain_data[-1]] + k_fold[key], alpha=0.4)
                # add to accumulator
                history_acc.append(comp_data)
        # calculate mean and standart deviation for each epoch
        mean = np.mean(history_acc, axis=0)
        std = np.std(history_acc, axis=0)
        # plot bar that shows mean results and their std
        error_line = ax.errorbar(range(len(mean)), mean, std, color='r', elinewidth=0.6)
        # final mean accuracy
        final_mean = ax.axhline(y=mean[-1], color='g')
        # pretrain epoch limit line
        last_pre_epoch = ax.axvline(x=len(pretrain_res[i][n][key])-1, color='c')
        # add last pretrain and last training epoch to subject accumulators
        print("LAST PRETRAIN INDEX:", len(pretrain_res[i][n][key])-1)
        pretrain_sub_acc.append(mean[len(pretrain_res[i][n][key])-1])
        train_sub_acc.append(mean[-1])
        # add legend to ax if it is first
        if i==0:
            ax.legend([error_line, final_mean, last_pre_epoch], ["mean and standartdeviation", "final mean acc.", "last pretrain epoch"], loc='upper left')
    plt.savefig(f'{figure_title}.png', bbox_inches='tight')
    return pretrain_sub_acc, train_sub_acc