import data_preprocessing as dp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d


# standardizing data
def z_score(x):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    xz = ss.fit_transform(x.T).T
    return xz


# trial response data
def trial_response(trials, start_action, end_action, standardised=False):
    data = np.vstack([t[:, start_action:-end_action].mean(axis=1) for t in trials]).T
    if standardised:
        data = z_score(data)
    return data


# trial average data
def trial_average(trials, t_type_ind, standardised=False):
    trial_averages = []
    for ind in t_type_ind:
        trial_averages.append(np.array(trials)[ind].mean(axis=0))
    data = np.hstack(trial_averages)
    if standardised:
        data = z_score(data)
    return data

# trial concatenated data
def trial_concat(trials, standardised=False):
    data = np.vstack([t[:] for t in trials]).T
    if standardised:
        data = z_score(data)
    return data