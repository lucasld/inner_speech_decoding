import os
import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import scipy.stats
import datetime
import os
import sklearn

import data_preprocessing as dp
from models.classifiers import EEGNet
from classify import pretrain_tester, kfold_training


# Hyperparameters
EPOCHS = 10
SUBJECT_S = range(1,11)
DROPOUT = 0.4
KERNEL_LENGTH = 64
N_CHECKS = 10
BATCH_SIZE = 10
PRETRAIN_EPOCHS = 35
MODE = 'pretrained'
FREEZE_LAYERS = []


def pretrained_all_classes(subject, train_subjects=range(1,11)):
    """Determine the performance of models to predict the inner speech data of
    one subject. The models are pretrained on all subjects 3 conditions and the
    two non-inner speech conditions of the subject in question.

    :param subject: the subject whose inner speech data should be predicted
    :type subject: integer from 1 to 10
    :param freeze_layers: all indices of layers that should be frozen after
        pretraining the network
    :type freeze_layers: list of integers
    :return: the history of the training and pretraining
    :rtype: list of dictonaries for every model that was created; N_CHECKS * 4
    """
    tf.keras.backend.clear_session()
    print(f"TESTING SUBJECT {subject}")
    # load all subjects individually
    subjects_data_collection = [dp.load_data(subjects=[s], filter_action=True) for s in range(1,11)]
    ###### INNER SPEECH SUBJECT DATA
    # collect subject's data and events
    subject_data_is, subject_events_is = subjects_data_collection[subject - 1]
    # save memory by converting from 64bit to 32bit floats
    subject_data_is = subject_data_is.astype(np.float32)
    # filter out only the inner speech condition
    subject_data_is, subject_events_is = dp.choose_condition(subject_data_is,
                                                             subject_events_is,
                                                             'inner speech')
    # select the column containing directions (up, down, left, right)
    subject_events_is = subject_events_is[:, 1]
    # one-hot event data 
    subject_events_is = np_utils.to_categorical(subject_events_is, 4)
    # zscore normalize the data
    subject_data_is = scipy.stats.zscore(subject_data_is, axis=2)
    # reshape
    subject_data_is = subject_data_is.reshape(*subject_data_is.shape, 1)
    
    ###### PRETRAIN DATA
    # collect pretrain data
    pretrain_data = [subjects_data_collection[i-1][0] for i in train_subjects if i != subject]
    pretrain_events = [subjects_data_collection[i-1][1] for i in train_subjects if i != subject]
    # append all non 'inner-speech'-conditions from subject 8
    for cond in ['pronounced speech', 'visualized condition']:
        data_sub_non_is, events_sub_non_is = dp.choose_condition(*subjects_data_collection[subject - 1], cond)
        # add the subjects non inner speech data to rest of the pretrain data
        pretrain_data.append(data_sub_non_is)
        pretrain_events.append(events_sub_non_is)
    # concatenate everything
    pretrain_data = np.concatenate(pretrain_data, axis=0)
    pretrain_events = np.concatenate(pretrain_events, axis=0)
    # same preprocessing as for the subjects inner speech data commented above
    pretrain_data = pretrain_data.astype(np.float32)
    pretrain_events = pretrain_events[:, 1]
    pretrain_events = np_utils.to_categorical(pretrain_events, num_classes=4)
    pretrain_data = scipy.stats.zscore(pretrain_data, axis=2)
    pretrain_data = pretrain_data.reshape(*pretrain_data.shape, 1)
    # create train and val tf.datasets
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    pt_train_ds = tf.data.Dataset.from_tensor_slices((pretrain_data, pretrain_events)).with_options(options)
     # turn subject data into tf.Dataset to use in pretraining validation
    pt_val_ds = tf.data.Dataset.from_tensor_slices((subject_data_is[:50], subject_events_is[:50])).with_options(options)
    # cache, shuffle, batch, prefetch
    pt_train_ds = dp.preprocessing_pipeline(pt_train_ds, batch_size=BATCH_SIZE)
    pt_val_ds = dp.preprocessing_pipeline(pt_val_ds, batch_size=BATCH_SIZE)
    # k-fold test EEGNet on provided data, repeat N_CHECK times
    pretrain_history, train_history = pretrain_tester(
        pt_train_ds, pt_val_ds, subject_data_is, subject_events_is,
        N_CHECKS, PRETRAIN_EPOCHS, EPOCHS, BATCH_SIZE, FREEZE_LAYERS,
        DROPOUT, KERNEL_LENGTH
    )
    # save progress
    print("Parameter Test Done")
    print("Average Accuracy:", np.mean([h['val_accuracy'][-1] for h in np.concatenate(train_history)]))
    print("Parameters")
    print("EPOCHS:", EPOCHS)
    print("SUBJECT:", subject)
    print("DROPOUT", DROPOUT)
    print("KERNEL_LENGTH", KERNEL_LENGTH)
    print("N_CHECKS", N_CHECKS)
    print("BATCH SIZE", BATCH_SIZE)
    #print(history_accumulator)
    return pretrain_history, train_history


def no_pretrain_inner_speech(subject):
    """This function aims at training a model without pretraining by training
    only on the inner speech condition of a sigle subject

    :return: metric history for every of the n k-folds
    :rtype: list of dictonaries
    """
    ###### DATA
    data, events = dp.load_data(subjects=[subject], filter_action=True)
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
    data = scipy.stats.zscore(data, axis=2)
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
        history_accumulator += history
    print(history_accumulator)
    print("Subject", subject, "  Mean Accuracy:", np.mean([h['val_accuracy'][-1] for h in history_accumulator]))
    return history_accumulator


if __name__ == '__main__':
    # read in command line options
    opts, _ = getopt.getopt(sys.argv[1:],"e:s:d:k:n:b:p:t:m:f:")
    now = datetime.datetime.now()
    title = f"{now.strftime('%A')}_{now.hour}_{str(now.minute).zfill(2)}"
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
        if name == '-s': SUBJECT_S = list(map(int, arg.split(',')))
        if name == '-d': DROPOUT = float(arg)
        if name == '-k': KERNEL_LENGTH = int(arg)
        if name == '-n': N_CHECKS = int(arg)
        if name == '-b': BATCH_SIZE = int(arg)
        if name == '-p': PRETRAIN_EPOCHS = int(arg)
        if name == '-m': MODE = arg
        if name == '-t': title = arg
        if name == '-f': FREEZE_LAYERS = list(map(int, arg.split(',')))
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

    # accumulators to track pretrain and train history
    pretrain_acc = []
    train_acc = []
    for subject in SUBJECT_S:
        # determine performance of a model that is pretrained on all the data
        # except inner speech data of one subject
        if MODE == 'pretrained':
            print("Pretrained Model!")
            pretrain_history, subject_history = pretrained_all_classes(subject, train_subjects=[subject])
        elif MODE == 'no_pretrain':
            print("NO Pretraining!")
            pretrain_history = []
            subject_history = no_pretrain_inner_speech(subject)
        # create directory if not already existing
        try:
            os.mkdir(f'./{title}')
        except:
            print(f"creating {title} not working")
        # write results to file
        with open(f'./{title}/results.txt', 'a') as f:
            f.write(f"\nSubject: {subject}\nEpochs: {EPOCHS},\
                    Pretrain Epochs: {PRETRAIN_EPOCHS},\
                    Batch Size: {BATCH_SIZE}, N Checks: {N_CHECKS},\
                    Dropout: {DROPOUT}\n\
                    Pretrain History: {pretrain_history}\n\
                    Subject History: {subject_history}\n\n")
        f.close()
        # add history to accumulator
        pretrain_acc.append(pretrain_history)
        train_acc.append(subject_history)
