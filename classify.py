from models.eegnet import EEGNet
import data_preprocessing as dp
import numpy as np
import tensorflow as tf
import sklearn


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


def kfold_training(data, labels, model_provided, batch_size, epochs, k=4):
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
        print(f"{k_i+1} of {k} starting...")
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
            model = tf.keras.models.load_model(model_provided) if type(model_provided) is str else tf.keras.models.clone_model(model_provided)
        # fit model to k-folded data
        hist = model.fit(dataset_train, epochs=epochs, verbose=1, validation_data=dataset_test)
        del model
        # add metric history to accumulator
        k_history.append(hist.history)
    return k_history


def pretrain_tester(pretrain_dataset, pretrain_val_dataset,
                   train_data, train_events, n_checks, pretrain_epochs, epochs,
                   batch_size, freeze_layers, dropout, kernel_length):
    """Pretrain and 'post-pre-train' a model.

    :param pretrain_dataset: dataset to pretrain the model with
    :type pretrain_dataset: tf.data.Dataset
    """
    ###### PRETRAIN AND TRANSFER LEARNING N_CHECKS TIMES
    pretrain_history_accumulator = []
    train_history_accumulator = []
    for n in range(n_checks):
        print(f"{n} of {n_checks}!")
        ###### PRETRAIN MODEL
        print("Pretraining...")
        tf.debugging.set_log_device_placement(True)
        gpus = tf.config.list_logical_devices('GPU')
        # tensorflows mirrored strategy adds support to do synchronous distributed
        # training on multiple GPU's
        mirrored_strategy = tf.distribute.MirroredStrategy(gpus)
        with mirrored_strategy.scope():
            # create EEGNet (source: https://github.com/vlawhern/arl-eegmodels)
            model_pretrain = EEGNet(nb_classes=4, Chans=train_data.shape[1],
                                    Samples=train_data.shape[2], dropoutRate=dropout,
                                    kernLength=kernel_length, F1=8, D=2, F2=16, dropoutType='Dropout')
            # adam optimizer
            optimizer = tf.keras.optimizers.Adam()
        # compile model
        model_pretrain.compile(loss='categorical_crossentropy',
                                optimizer=optimizer,
                                metrics=['accuracy'])
        # fit model to pretrain data
        pretrain_history = model_pretrain.fit(pretrain_dataset, epochs=pretrain_epochs,
                                            verbose=1, validation_data=pretrain_val_dataset)
        # append pretrain history to accumulator
        pretrain_history_accumulator.append(pretrain_history.history)
        # save pretrained model so it can be used for transfer learning
        path = './models/saved_models/pretrained_model01'
        print("FREEZE?")
        for freeze_index in freeze_layers:
            # function to get trainable parameters
            trainable_params = lambda: np.sum([np.prod(v.get_shape()) for v in model_pretrain.trainable_weights])
            print("trainable parameters before freezing:", trainable_params())
            model_pretrain.layers[freeze_index].trainable = False
            print("after:", trainable_params())
        model_pretrain.save(path)
        del model_pretrain
        print("Pretraining Done")
        # TRANSFER LEARNING
        # kfold testing of transfer learning
        k_history = kfold_training(train_data, train_events, path, batch_size, epochs)
        # add kfold metric-history
        train_history_accumulator.append(k_history)
        print("\n\nN: ", n, "     ######################\n")
        print("Mean for K Folds:", np.mean([h['val_accuracy'][-1] for h in k_history]))
        print("New Total Mean:", np.mean([h['val_accuracy'][-1] for h in np.concatenate(train_history_accumulator)]))
    return pretrain_history_accumulator, train_history_accumulator