# https://pietromarchesi.net/pca-neural-data.html

import data_preprocessing as dp
import pca_preprocessing as pp
import pca_plotting as ppl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# load data
data, events = dp.load_data(subjects=[1])
data, events = dp.choose_condition(data, events, 'inner speech')
#data, events = dp.filter_interval((data[1], events[1]), [1, 3.5], 256)

rec_freq = 256
pre_action = 1 * rec_freq
post_action = 1 * rec_freq

trials = data
trial_type = events[:,1]
trial_types = [0, 1, 2, 3]
start_action = 1
end_action = 3.5
time = np.linspace(1, 3.5 , trials[0].shape[1])
trial_size = trials[0].shape[1]
channels = trials[0].shape[0]

# list of arrays containing the indices of trials of each type (t_type_ind[0] contains the
# indices of trials of type trial_types[0])
t_type_ind = [np.argwhere(np.array(trial_type) == t_type)[:, 0] for t_type in trial_types]

print('Number of trials: {}'.format(len(trials)))
print('Types of trials (orientations): {}'.format(trial_types))
print('Dimensions of single trial array (# channels by # time points): {}'.format(trials[0].shape))
print('Trial types (orientations): {}'.format(trial_types))
print('Trial type of the first 3 trials: {}'.format(trial_type[0:3]))


n_components = 0.98

# Trial Response PCA
Xr = pp.trial_response(trials, pre_action, post_action, standardised=True)
pca1 = PCA(n_components=n_components)
Xp = pca1.fit_transform(Xr.T).T
print(Xr.shape)
print(Xp.shape)

ppl.variance(pca1)
ppl.scatter(Xp, t_type_ind)

# trial averaged PCA
Xa = pp.trial_average(trials, t_type_ind, standardised=True)
pca2 = PCA(n_components=n_components)
Xa_p = pca2.fit_transform(Xa.T).T

print(Xa.shape)
print(Xa_p.shape)
ppl.variance(pca2)
ppl.scatter(Xa_p, t_type_ind)
ppl.plot(Xa_p, 3, trial_size, time)

# trial concatenated
Xc = pp.trial_concat(trials, standardised=True)
pca3 = PCA(n_components=n_components)
Xcp = pca3.fit_transform(Xc.T).T
print(Xc.shape)
print(Xcp.shape)

ppl.variance(pca3)
ppl.scatter(Xcp, t_type_ind)

