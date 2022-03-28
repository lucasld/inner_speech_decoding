from asyncio import events
from curses.ascii import SUB
import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import sklearn
import scipy.stats
from perlin_numpy import generate_perlin_noise_3d
from numba import cuda 
import nvsmi

import data_preprocessing as dp
from models.classifiers import EEGNet

# Hyperparameters
EPOCHS = 40
SUBJECT_S = range(1,11)
DROPOUT = 0.8
KERNEL_LENGTH = 64
N_CHECKS = 20
BATCH_SIZE = 40


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


def kfold_training_pretrained(data, labels, path, k=4):
    """ This function is used to test the effectifness of pretraining the model.
    First the model will be trained on the other subjects, then the model is
    trained on one previously unseen subject. Testing of the models accuracy
    will also only rely on this one subject. (transfer-learning?)
    
    data, labels are correspond to the subject to be tested
    """
    data, labels = sklearn.utils.shuffle(data, labels)
    # create k data and label splits
    X = []
    Y = []
    for i in range(k):
        n = data.shape[0]
        X.append(data[int(n/k * i):int(n/k * i + n/k)])
        Y.append(labels[int(n/k * i):int(n/k * i + n/k)])
    
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
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        dataset_train = dataset_train.with_options(options)
        dataset_train = dp.preprocessing_pipeline(dataset_train, batch_size=BATCH_SIZE)
        # test dataset
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).with_options(options)
        dataset_test = dp.preprocessing_pipeline(dataset_test, batch_size=BATCH_SIZE)
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = tf.keras.models.load_model(path)
            class_weights = {0:1, 1:1, 2:1, 3:1}
            hist = model.fit(dataset_train, epochs = EPOCHS, verbose = 1,
                             validation_data=dataset_test,
                             class_weight=class_weights)
        k_history.append(hist.history)
    return k_history

# + testen obs so geht +
# subject training vereinzeln

# dataset preloaden und nicht bei jedem subject erneut
# history plotten / speichern
# model loading in kfold_training verbessern mit cloning erweitern


def subject_train_test_average(subject, complete_dataset):
    tf.keras.backend.clear_session()
    print(f"TESTING SUBJECT {subject}")
    # subject data
    subject_data_is, subject_events_is = complete_dataset[subject - 1]
    subject_data_is = subject_data_is.astype(np.float32)
    subject_data_is, subject_events_is = dp.choose_condition(subject_data_is,
                                                             subject_events_is,
                                                             'inner speech')
    subject_events_is = subject_events_is[:, 1]
    subject_events_is = np_utils.to_categorical(subject_events_is, num_classes=4)
    subject_data_is = scipy.stats.zscore(subject_data_is, axis=1)
    # reshape
    subject_data_is = subject_data_is.reshape(*subject_data_is.shape, 1)
    
    # pretrain data
    pretrain_data = np.concatenate([data for i, (data, target) in enumerate(complete_dataset) if i != subject-1], axis=0)
    pretrain_events = np.concatenate([target for i, (data, target) in enumerate(complete_dataset) if i != subject-1], axis=0)
    print(pretrain_data.shape)
    # append all non 'inner-speech'-conditions from subject 8
    subject_data_all, subject_events_all = complete_dataset[subject - 1]
    for cond in ['pronounced speech', 'visualized condition']:
        data_subject_nis, events_subject_nis = dp.choose_condition(subject_data_all, subject_events_all, cond)
        pretrain_data = np.append(pretrain_data, data_subject_nis, axis=0)
        pretrain_events = np.append(pretrain_events, events_subject_nis, axis=0)
    pretrain_data = pretrain_data.astype(np.float32)
    # filter relevant column from events
    pretrain_events = pretrain_events[:, 1]
    # one hot events
    pretrain_events = np_utils.to_categorical(pretrain_events, num_classes=4)
    # normlize data
    pretrain_data = scipy.stats.zscore(pretrain_data, axis=1)
    # reshape
    pretrain_data = pretrain_data.reshape(*pretrain_data.shape, 1)

    # Pretrain Model ----------
    print("Pretraining...")
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model_pretrain = EEGNet(nb_classes=4, Chans=pretrain_data.shape[1],
                                Samples=pretrain_data.shape[2], dropoutRate=DROPOUT,
                                kernLength=KERNEL_LENGTH, F1=8, D=2, F2=16, dropoutType='Dropout')
        optimizer = tf.keras.optimizers.Adam()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = tf.data.Dataset.from_tensor_slices((pretrain_data, pretrain_events)).with_options(options)
    dataset = dp.preprocessing_pipeline(dataset, batch_size=BATCH_SIZE)
    model_pretrain.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])
    class_weights = {0:1, 1:1, 2:1, 3:1}
    model_pretrain.fit(dataset, epochs=EPOCHS, 
                        verbose = 1, class_weight = class_weights)
    print("Pretraining Done")
    path = './models/saved_models/pretrained_model01'
    model_pretrain.save(path)
    # ------------------------
    # accumulate all training accs and losses
    history_accumulator = []
    for n in range(N_CHECKS):
        # clear gpu memory        
        #device = cuda.get_current_device()
        #device.reset()
        # train k folds
        print(subject_data_is.shape, subject_events_is.shape)
        k_history = kfold_training_pretrained(subject_data_is,
                                              subject_events_is, path)
        history_accumulator += k_history
        print("N: ", n, "     ######################\n\n")
        print("Mean for K Folds:", np.mean([h['val_accuracy'][-1] for h in k_history]))
        print("New Total Mean:", np.mean([h['val_accuracy'][-1] for h in history_accumulator]))
    # save progress
    print("Parameter Test Done")
    print("Average Accuracy:", np.mean([h['val_accuracy'][-1] for h in history_accumulator]))
    print(history_accumulator)
    print("Parameters")
    print("EPOCHS:", EPOCHS)
    print("SUBJECT:", subject)
    print("DROPOUT", DROPOUT)
    print("KERNEL_LENGTH", KERNEL_LENGTH)
    print("N_CHECKS", N_CHECKS)
    print("BATCH SIZE", BATCH_SIZE)
    f = open("results.txt", "a")
    f.write('\n'.join([f'Subject {subject}', f"Average Accuracy: {np.mean([h['val_accuracy'][-1] for h in history_accumulator])}", f'{history_accumulator}', "\n\n\n"]))
    f.close()
    #subject_history = history_accumulator
    return history_accumulator


if __name__ == '__main__':
    opts, _ = getopt.getopt(sys.argv[1:],"e:s:d:k:n:b:")
    print(opts)
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
    
    # load all subjects individually
    subjects_data_collection = [dp.load_data(subjects=[s]) for s in SUBJECT_S]

    for subject in SUBJECT_S:
        # option 1: execute code with extra process
        subject_history = subject_train_test_average(subject, subjects_data_collection)
        gpu1 = list(nvsmi.get_gpus())[0]
        print("FREE MEMORY:", gpu1.mem_util)
        print("USED MEMORY:", gpu1.mem_free)
        #device = cuda.get_current_device()
        #device.reset()
        for h in subject_history:
            plt.plot(h['val_accuracy'])
        plt.savefig(f'subject_{subject}_kfold_accuracies.png')
        plt.clf()
