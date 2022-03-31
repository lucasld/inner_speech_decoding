
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
    only using the inner speech condition. As always we are only interested in
    inner speech predicitive performance.

    :return: metric history for every of the n k-folds
    :rtype: list of dictonaries
    """
    ###### DATA
    data, events = dp.load_data(filter_action=True)
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
    print("Data Prepared.")
    ###### MODEL
    gpus = tf.config.list_logical_devices('GPU')
    mirrored_strategy = tf.distribute.MirroredStrategy(gpus)
    with mirrored_strategy.scope():
        # create EEGNet (source: https://github.com/vlawhern/arl-eegmodels)
        model = EEGNet(nb_classes=4, Chans=data.shape[1],
                                Samples=data.shape[2], dropoutRate=DROPOUT,
                                kernLength=KERNEL_LENGTH, F1=8, D=2, F2=16,
                                dropoutType='Dropout')
        # adam optimizer
        optimizer = tf.keras.optimizers.Adam()
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.build(input_shape=(BATCH_SIZE, *data.shape[1:]))
    path = './models/saved_models/no_pretrain_inner_speech'
    model.save(path)
    del model
    ###### KFOLD TRAINING
    history_accumulator = []
    for _ in range(N_CHECKS):
        history = kfold_training(data, events, path, BATCH_SIZE, EPOCHS)
        history_accumulator.append(history)
    return history_accumulator


def pretrain_non_inner_speech():
    """This function aims at pretraining a model on pronounced speech and
    visualized condition data and then evaluating the effectiveness of transfer
    learning to the inner speech condition (k-fold)

    :return: metric history for every of the n k-folds and the pretraining
    :rtype: two lists of dictonaries   
    """
    # load all data
    data, events = dp.load_data(filter_action=True)
    # shuffle data and labels
    data, events = sklearn.utils.shuffle(data, events)
     # save memory by converting from 64bit to 32bit floats
    data = data.astype(np.float32)
    # reshape
    data = data.reshape(*data.shape, 1)
    # pretrain data
    data_pronounced_speech, events_pronounced_speech = dp.choose_condition(data, events, 'pronounced speech')
    data_visualized_cond, events_visualized_cond = dp.choose_condition(data, events, 'visualized condition')
    data_pretrain = np.concatenate([data_pronounced_speech, data_visualized_cond], axis=0)
    events_pretrain = np.concatenate([events_pronounced_speech, events_visualized_cond], axis=0)
    # select the column containing directions (up, down, left, right)
    events_pretrain = events_pretrain[:, 1]
    # one-hot event data 
    events_pretrain = np_utils.to_categorical(events_pretrain, 4)
    # zscore normalize the data
    data_pretrain = scipy.stats.zscore(data_pretrain, axis=1)
    # inner speech data
    data_inner_speech, events_inner_speech = dp.choose_condition(data, events, 'inner speech')
    # convert to tf.Dataset
    dataset_pretrain = tf.data.Dataset.from_tensor_slices((data_pretrain, events_pretrain))
    dataset_inner_speech = tf.data.Dataset.from_tensor_slices((data_inner_speech, events_inner_speech))

    ###### PRETRAINING
    print("Pretraining...")
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    # tensorflows mirrored strategy adds support to do synchronous distributed
    # training on multiple GPU's
    mirrored_strategy = tf.distribute.MirroredStrategy(gpus)
    with mirrored_strategy.scope():
        # create EEGNet (source: https://github.com/vlawhern/arl-eegmodels)
        model_pretrain = EEGNet(nb_classes=4, Chans=data.shape[1],
                                Samples=data.shape[2], dropoutRate=DROPOUT,
                                kernLength=KERNEL_LENGTH, F1=8, D=2, F2=16, dropoutType='Dropout')
        # adam optimizer
        optimizer = tf.keras.optimizers.Adam()
    # compile model
    model_pretrain.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])
    # fit model to pretrain data
    pretrain_history = model_pretrain.fit(data_pretrain, epochs=PRETRAIN_EPOCHS,
                                          verbose=1, validation_data=dataset_inner_speech)
    # save pretrained model so it can be used for transfer learning
    path = './models/saved_models/pretrained_subject_independent'
    model_pretrain.save(path)
    del model_pretrain
    print("Pretraining Done")

    ###### TRANSFER LEARNING
    history_accumulator = []
    for n in range(N_CHECKS):
        # kfold testing of transfer learning
        k_history = kfold_training(data_inner_speech, events_inner_speech, BATCH_SIZE, EPOCHS, path)
        # add kfold metric-history
        history_accumulator += k_history
        print("\n\nN: ", n, "     ######################\n")
        print("Mean for K Folds:", np.mean([h['val_accuracy'][-1] for h in k_history]))
        print("New Total Mean:", np.mean([h['val_accuracy'][-1] for h in history_accumulator]))
    
    # save progress
    print("Parameter Test Done")
    print("Average Accuracy:", np.mean([h['val_accuracy'][-1] for h in history_accumulator]))
    print("Parameters")
    print("EPOCHS:", EPOCHS)
    print("DROPOUT", DROPOUT)
    print("KERNEL_LENGTH", KERNEL_LENGTH)
    print("N_CHECKS", N_CHECKS)
    print("BATCH SIZE", BATCH_SIZE)
    print(history_accumulator)
    return pretrain_history.history, history_accumulator


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
        print("NO PRETRAINING!")
        history = no_pretrain_inner_speech()
        pretrain_history = []
    elif MODE == 'pretrain':
        print("PRETRAINING")
        pretrain_history, history = pretrain_non_inner_speech()
    
    try:
        os.mkdir(f'./{title}')
    except:
        print(f"creating {title} not working")
    # plot all subject's inter-training results
    f = open(f'./{title}/results.txt', 'a')
    f.write(f"\n\n{history}")
    f.close()
    # plot all results
    plot_inter_train_results(history, f'./{title}/all_subjects', pretrain_res=pretrain_history)
