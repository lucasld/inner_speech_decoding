import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
from asyncio import events
from curses.ascii import SUB
import sys, getopt
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import sklearn
import scipy.stats
from perlin_numpy import generate_perlin_noise_3d
from numba import cuda 
import nvsmi
import datetime
import os
tf.autograph.set_verbosity(3)
import logging
logging.getLogger('tensorflow').disabled = True
tf.get_logger().setLevel('INFO')

import data_preprocessing as dp
from models.classifiers import EEGNet
from utilities import plot_inter_train_results


def augment_pipe(data, events, noise):
    aug_data = data + np.random.normal(0, np.random.rand() * 2, data.shape) # 100_000
    for i in range(aug_data.shape[0]):
        if np.random.rand() < 0.5: aug_data[i] = np.fliplr(aug_data[i])
        if np.random.rand() < 0.5: aug_data[i] = np.flipud(aug_data[i])
        # salt pepper
        p = np.random.rand() * 0.4  # 0.4
        r = np.random.rand(*aug_data[i].shape)
        u, l = r > (1 - p/2), r < p/2
        aug_data[i][u] = 1#np.max(aug_data[i])
        aug_data[i][l] = -1#np.min(aug_data[i])
    
    aug_data += 20 * noise.reshape((*noise.shape, 1))[:data.shape[0]]
    
    return aug_data, events

"""
def kfold_training(data, labels, k=4):
    data, labels = sklearn.utils.shuffle(data, labels)
    # create k data and label splits
    X = []
    Y = []
    # create perlin noise
    noise = generate_perlin_noise_3d(
        (data.shape[0], 128, 1200), (5, 4, 4), tileable=(True, False, False) #4, 4
    )
    # shuffle noise
    noise = np.random.default_rng().permutation(noise)[:, :, :data.shape[2]]

    for i in range(k):
        n, _, _ = data.shape
        X.append(data[int(n/k * i):int(n/k * i + n/k)])
        Y.append(labels[int(n/k * i):int(n/k * i + n/k)])
    
    kfold_acc = []
    for k_i in range(k):
        # concat k-1 splits
        X_train = np.concatenate([d for j, d in enumerate(X) if j != i])
        Y_train = np.concatenate([d for j, d in enumerate(Y) if j != i])
        X_test = X[i]
        Y_test = Y[i]
        kernels, chans, samples = 1, data.shape[1], data.shape[2]
        # reshape
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
        model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
                dropoutRate = DROPOUT, kernLength = KERNEL_LENGTH, F1 = 8, D = 2, F2 = 16, 
                dropoutType = 'Dropout')
        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                    metrics = ['accuracy'])
        class_weights = {0:1, 1:1, 2:1, 3:1}
        # train, in each epoch train data is augmented
        for _ in range(EPOCHS):
            X_aug, Y_aug = augment_pipe(X_train, Y_train, noise)
            model.fit(X_aug, Y_aug, batch_size = BATCH_SIZE, epochs = 1, 
                      verbose = 1, validation_data=(X_test, Y_test), class_weight = class_weights)
        # test trained model
        probs = model.predict(X_test)
        preds = probs.argmax(axis = -1)  
        acc = np.mean(preds == Y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))
        kfold_acc.append(acc)
    return kfold_acc
"""

def kfold_training(data, labels, model_path, batch_size, epochs, k=4):
    """K-Fold train and test a model on data and labels.

    :param data: models input data
    :type data: numpy array
    :param labels: labels corresponding to data
    :type labels: numpy array
    :param model_path: path of the saved model to be used (pretrained or not)
    :type model_path: string
    :param batch_size: datasets batch size
    :type batch_size: integer
    :param epochs: number of epochs for which to train the model
    :type epochs: integer
    :param k: number of folds the data should be split into when training and
        testing, defaults to 4
    :type k: integer, optional
    :return: the history of the k folds
    :rtype: list containing dictonaries saving the different metrics
    """
    # shuffle data and labels
    data, labels = sklearn.utils.shuffle(data, labels)
    # create k data and label splits
    X = []
    Y = []
    for i in range(k):
        n = data.shape[0]
        X.append(data[int(n/k * i):int(n/k * i + n/k)])
        Y.append(labels[int(n/k * i):int(n/k * i + n/k)])
    # list accumulating every folds metrics 
    k_history = []
    for k_i in range(k):
        tf.keras.backend.clear_session()
        # concat k-1 splits
        X_train = np.concatenate([d for j, d in enumerate(X) if j != k_i])
        Y_train = np.concatenate([d for j, d in enumerate(Y) if j != k_i])
        X_test = X[k_i]
        Y_test = Y[k_i]
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # train dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).with_options(options)
        dataset_train = dp.preprocessing_pipeline(dataset_train, batch_size=batch_size)
        # test dataset
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).with_options(options)
        dataset_test = dp.preprocessing_pipeline(dataset_test, batch_size=batch_size)
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.list_logical_devices('GPU')
        mirrored_strategy = tf.distribute.MirroredStrategy(gpus)
        with mirrored_strategy.scope():
            # load pretrained model
            model = tf.keras.models.load_model(model_path)
        # fit model to k-folded data
        hist = model.fit(dataset_train, epochs=epochs, verbose=1, validation_data=dataset_test)
        del model
        # add metric history to accumulator
        k_history.append(hist.history)
    return k_history