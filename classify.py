import sys, getopt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import sklearn
import scipy.stats
from perlin_numpy import generate_perlin_noise_3d


import data_preprocessing as dp
from models.classifiers import EEGNet


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
    
    aug_data += 10 * noise.reshape((*noise.shape, 1))[:data.shape[0]]
    
    return aug_data, events


def kfold_training(data, labels, k=4):
    data, labels = sklearn.utils.shuffle(data, labels)
    # create k data and label splits
    X = []
    Y = []
    # create perlin noise
    noise = generate_perlin_noise_3d(
        (data.shape[0], 128, 1200), (5, 4, 4), tileable=(True, False, False)
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
            model.fit(X_aug, Y_aug, batch_size = 12, epochs = 1, 
                      verbose = 1, validation_data=(X_test, Y_test), class_weight = class_weights)
        # test trained model
        probs = model.predict(X_test)
        preds = probs.argmax(axis = -1)  
        acc = np.mean(preds == Y_test.argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))
        kfold_acc.append(acc)
    
    return kfold_acc


if __name__ == '__main__':
    EPOCHS = 40
    SUBJECT = 8
    DROPOUT = 0.8
    KERNEL_LENGTH = 64
    N_CHECKS = 20

    opts, _ = getopt.getopt(sys.argv[1:],"e:s:d:k:n:")
    print(opts)
    for name, arg in opts:
        """ # Python 3.10
        match name:
            case 'e': EPOCHS = int(arg)
            case 's': SUBJECT = int(arg)
            case 'd': DROPOUT = float(arg)
            case 'k': KERNEL_LENGTH = int(arg)
            case 'n': N_CHECKS = int(arg)
        """
        # < Python 3.10
        if name == 'e': EPOCHS = int(arg)
        if name == 's': SUBJECT = int(arg)
        if name == 'd': DROPOUT = float(arg)
        if name == 'k': KERNEL_LENGTH = int(arg)
        if name == 'n': N_CHECKS = int(arg)
    # load data
    subject_data, subject_events = dp.load_data(subjects=[SUBJECT])
    # choose condition
    subject_data, subject_events = dp.choose_condition(subject_data, subject_events, 'inner speech')
    # filter relevant column from events
    subject_events = subject_events[:,1]
    # one hot events
    subject_events = np_utils.to_categorical(subject_events, num_classes=4)
    # normlize data
    subject_data = scipy.stats.zscore(subject_data, axis=1)
    # k fold test n_checks times to average accuracy over many different splits
    acc_accumulator = []
    for n in range(N_CHECKS):
        kacc = kfold_training(subject_data, subject_events)
        acc_accumulator += kacc
        print("N: ", n, "     ######################\n\n")
        print("Mean for K Folds:", np.mean(kacc))
        print("New Total Mean:", np.mean(acc_accumulator))
    # save progress
    print("Parameter Test Done")
    print("Average Accuracy:", np.mean(acc_accumulator))
    print(acc_accumulator)
    print("Parameters")
    print("EPOCHS:", EPOCHS)
    print("SUBJECT:", SUBJECT)
    print("DROPOUT", DROPOUT)
    print("KERNEL_LENGTH", KERNEL_LENGTH)
    print("N_CHECKS", N_CHECKS)
 



