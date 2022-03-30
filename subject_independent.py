
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
from classify import kfold_training

# Hyperparameters
EPOCHS = 40
MODE = 'no_pretrain'
DROPOUT = 0.8
KERNEL_LENGTH = 64
N_CHECKS = 20
BATCH_SIZE = 40
PRETRAIN_EPOCHS = -1


def no_pretrain_inner_speech():
    """This function aims at training a model without pretraining by training
    on a set of conditions. As always we are only interested in inner speech predicitive performance.

    :return: metric history for every of the n k-folds
    :rtype: list of dictonaries
    """
    ###### MODEL
    # create EEGNet (source: https://github.com/vlawhern/arl-eegmodels)
    model = EEGNet(nb_classes=4, Chans=data.shape[1],
                            Samples=data.shape[2], dropoutRate=DROPOUT,
                            kernLength=KERNEL_LENGTH, F1=8, D=2, F2=16,
                            dropoutType='Dropout')
    # adam optimizer
    optimizer = tf.keras.optimizers.Adam()
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    path = './models/saved_models/no_pretrain_inner_speech'
    model.save(path)
    del model

    ###### DATA
    data, events = dp.load_data()
    # shuffle data and labels
    data, events = sklearn.utils.shuffle(data, events)
    # save memory by converting from 64bit to 32bit floats
    data = data.astype(np.float32)
    # filter out only the inner speech condition
    data, events = dp.choose_condition(data, events, 'inner speech')
    # select the column containing directions (up, down, left, right)
    events = events[:, 1]
    # one-hot event data 
    events = np_utils.to_categorical(events, 4)
    # zscore normalize the data
    data = scipy.stats.zscore(data, axis=1)
    # reshape
    data = data.reshape(*data.shape, 1)
    
    ###### KFOLD TRAINING
    kfold_training(data, events, path)



if __name__ == '__main__':
    # read in command line options
    opts, _ = getopt.getopt(sys.argv[1:],"e:m:d:k:n:b:p:t:")
    now = datetime.datetime.now()
    title =f"{now.strftime('%A')}_{now.hour}_{str(now.minute).zfill(2)}"
    for name, arg in opts:
        """ # Python 3.10
        match name:
            case '-e': EPOCHS = int(arg)
            case '-s': SUBJECT = int(arg)
            case '-d': DROPOUT = float(arg)
            case '-k': KERNEL_LENGTH = int(arg)
            case '-n': N_CHECKS = int(arg)
            case '-b': BATCH_SIZE = int(arg)
        """
        # < Python 3.10
        if name == '-e': EPOCHS = int(arg)
        if name == '-m': MODE = arg
        if name == '-d': DROPOUT = float(arg)
        if name == '-k': KERNEL_LENGTH = int(arg)
        if name == '-n': N_CHECKS = int(arg)
        if name == '-b': BATCH_SIZE = int(arg)
        if name == '-p': PRETRAIN_EPOCHS = int(arg)
        if name == '-t': title = arg
    # if pretrain epochs where not specified, pretrain epochs equal epochs
    if PRETRAIN_EPOCHS < 0: PRETRAIN_EPOCHS = EPOCHS
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if MODE == 'no_pretrain':
        history = no_pretrain_all_data()
    # plot all subject's inter-training results
    f = open(f'./{title}/results.txt', 'a')
    f.write(f"\n\n{history}")
    f.close()
    # plot all results
    plot_inter_train_results(history, f'./{title}/all_subjects')
