import data_preprocessing as dp

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# load data
data, events = dp.load_data(subjects=[1])
data, events = dp.choose_condition(data, events, 'inner speech')
# data preprocessing - normalization
data = data.astype(np.float16) * 100_000
data = dp.normalization(data)

print(data.shape)
# test PCa with a single trial from one participant
print(data[1].shape)
test, event = dp.filter_interval((data[1], events[1]), [1, 3.5], 128)
# PCA execution
pca = PCA(n_components=0.95, random_state=0)  #
feature = pca.fit_transform(test)
PCA_comp = pca.components_  # Basis matrix
print(feature.shape)

# Restore 0th data
plt.figure(figsize=(12, 2))
plt.plot(test[0], label="data")  # 0th data
plt.plot(np.dot(feature[0], PCA_comp).T, label="reconstruct")  # Restored 0th data
plt.legend()
plt.show()
