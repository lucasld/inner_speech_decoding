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
def trial_average(trials, t_type_ind, standardised=False, trial_types = [0, 1, 2, 3]):
    trial_averages = []
    trial_size = trials[0].shape[1]
    event = []
    for ind in t_type_ind:
        trial_averages.append(np.array(trials)[ind].mean(axis=0))
    data = np.hstack(trial_averages)
    for type in trial_types:
        event.append([type]*trial_size)
    event = np.hstack(event)
    if standardised:
        data = z_score(data)
    return data, np.array(event)

def flat_pca(data):
    flat_data = []
    for elem in data:
        flat_data.append(elem.reshape(len(elem[0]*len(elem[1]))))
    return flat_data
