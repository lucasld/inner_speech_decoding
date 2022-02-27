import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def variance(pca):
    total_var = pca.explained_variance_.sum()
    var_exp = [(v / total_var) * 100 for v in pca.explained_variance_]
    cumulative_var = [np.sum(var_exp[0:i]) for i in range(1, len(var_exp) + 1)]

    f, ax = plt.subplots(1, 1, figsize=[10, 5])
    ax.plot(range(len(var_exp)), var_exp, '-o', label='% variance explained', color=sns.xkcd_rgb["azure"])
    ax.plot(range(len(var_exp)), cumulative_var, '-o', label='Cumulative % variance explained',
            c=sns.xkcd_rgb["pinkish red"])
    ticks = ax.set_xticks(range(len(var_exp)))
    tick_lab = ax.set_xticklabels(range(1, len(var_exp) + 1))
    yticks = ax.set_yticks(np.arange(0, 110, 10))
    sns.despine(fig=f, right=True, top=True)
    xlab = ax.set_xlabel('Components')
    ax.grid(ls=':')
    l = ax.legend(frameon=False)
    f.show()


def add_stim_to_plot(ax, start_action, end_action):
    ax.axvspan(start_action, end_action, alpha=0.1,
               color='gray')
    ax.axvline(start_action, alpha=0.9, color='gray', ls='--')
    ax.axvline(end_action, alpha=0.9, color='gray', ls='--')

def scatter(pca_data, t_type_ind, trial_types = [0, 1, 2, 3]):
    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=[9, 4], sharey='row', sharex='row')
    for ax, proj in zip(axes, projections):
        for t, t_type in enumerate(trial_types):
            x = pca_data[proj[0], t_type_ind[t]]
            y = pca_data[proj[1], t_type_ind[t]]
            ax.scatter(x, y, s=25, alpha=0.8)
            ax.set_xlabel('PC {}'.format(proj[0] + 1))
            ax.set_ylabel('PC {}'.format(proj[1] + 1))
    sns.despine(fig=fig, top=True, right=True)
    fig.legend(trial_types)
    fig.show()


def plot(pca_data, i, trial_size, time, trial_types = [0, 1, 2, 3]):
    fig, axes = plt.subplots(1, i, figsize=[5*i, 2.8], sharey='row')
    for comp in range(i):
        ax = axes[comp]
        for kk, type in enumerate(trial_types):
            x = pca_data[comp, kk * trial_size:(kk + 1) * trial_size]
            x = gaussian_filter1d(x, sigma=3)
            ax.plot(time, x)
        #add_stim_to_plot(ax)
        ax.set_ylabel('PC {}'.format(comp + 1))
    axes[1].set_xlabel('Time (s)')
    sns.despine(fig=fig, right=True, top=True)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.legend(trial_types)
    fig.show()