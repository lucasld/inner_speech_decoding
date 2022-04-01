import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA
import data_preprocessing as dp

""" Data wrangling methods"""


def load_data(subjects=range(1, 11), time=[1, 3.5]):
    """ Create the initial pca data to train a network
    :param subjects: Number of subjects who's data will be prepared
    :type subjects: range of integer numbers from 1-11, standard is all subjects
    :param time: Time frame extracted from the whole eeg trial max: [0, 4.5]
    :type time: array of floats [start, end]
    :return dictionary of data points, events, scaler and pca with the experimental conditions as keys
    :rtype float, float, RobustScaler, PCA
    """
    data, events = dp.load_data(subjects=subjects)
    d = {}
    e = {}
    for condition in ['inner speech', 'pronounced speech', 'visualized condition']:
        temp_data, temp_events = dp.choose_condition(data, events, condition)
        # extract relevant columns
        temp_events = np.array(temp_events[:, 1])
        temp_data = temp_data[:, :, int(time[0] * 256):int(time[1] * 256)]
        # Rescale the data
        s = RobustScaler()
        temp_data = s.fit_transform(temp_data.reshape(-1, temp_data.shape[-1])).reshape(temp_data.shape)
        # save the cleaned data in dictionaries for further work
        e[condition] = temp_events
        d[condition] = temp_data

    # Prepare Pretraining Data
    pre_data = np.array(d['pronounced speech'])
    pre_data = np.concatenate([pre_data, np.array(d['visualized condition'])], axis=0)
    pre_events = e['pronounced speech']
    pre_events = np.concatenate([pre_events, e['visualized condition']], axis=0)
    pre_events = np.array(pre_events)

    # Prepare Training Data
    training_data = np.array(d['inner speech'])
    training_events = np.array(e['inner speech'])

    return (pre_data, pre_events), (training_data, training_events)


def data_preprocessing(x, y, pca_type, pca_components, pca_condition, batch_size=10, batched=True, fit_pca=None):
    """
    Preprocessing pipeline that applies the PCA on the x data before transforming x and y into a tensorflow dataset or
    a tensor tuple.

    :param x: eeg data
    :type x: array of type trials*X
    :param y: label data
    :type y: array of size trials
    :param pca_type: type of pca we want to apply to the data coded as integers.
                        1: pca on reshaped data
                        2: pca on the channels
                        3: pca on the timepoints
    :type pca_type: integer, optional
    :param pca_condition: additional parameter only used for pca_type 2 and 3.
                        0: pca is fit on the mean of all data-points
                        1: pca is fit onto each datapoint individually
    :type pca_condition: Integer, optional
    :param pca_components: Number of components of the pca
    :type pca_components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :param batch_size: Batchsize of the dataset returned
    :type batch_size: Integer, optional
    :param batched: Determines the output. If batched is true a dataset will be created,
        if batched is false a tuple is generated
    :type batched: Boolean
    :param fit_pca: Determines if pca should be applied with an already fit PCA
    :type fit_pca: sklearn.decomposit.PCA.fit()
    :return:
    """
    pca = None
    # Apply PCA on the eeg data
    if pca_type:
        x, pca = apply_pca(x, pca_type=pca_type, pca_components=pca_components, pca_condition=pca_condition,
                           fit_pca=fit_pca)
    # Prepare the tensorflow dataset
    if batched:
        data = tf.data.Dataset.from_tensor_slices((x, y))
        data = data.map(lambda image, label: (tf.cast(image, tf.float32), tf.one_hot(int(label), 4)))
        if pca_type != 1:
            data = data.map(lambda image, label: (tf.expand_dims(image, axis=-1), label))
        # cache the dataset
        data = data.cache()
        # shuffle, batch and prefetch the dataset
        data = data.shuffle(100)
        data = data.batch(batch_size)
        data = data.prefetch(10)
    else:
        x = tf.constant(x)
        y = tf.constant(y)

        x = tf.cast(x, tf.float32)
        y = tf.one_hot(y, 4)
        data = (x, y)

    # return preprocessed dataset
    return data, pca


""" Methods to apply PCA on EEG data"""


def apply_pca(data, pca_type=2, pca_condition=0, pca_components=0.98, fit_pca=None):
    """
    Method that applies the chosen type of pca on an eeg dataset
    :param data:eeg data
    :type data: array of shape trials, channel, time
    :param pca_type: type of pca we want to apply to the data coded as integers.
                        1: pca on reshaped data
                        2: pca on the channels
                        3: pca on the timepoints
    :type pca_type: integer, optional
    :param pca_condition: additional parameter only used for pca_type 2 and 3.
                        0: pca is fit on the mean of all data-points
                        1: pca is fit onto each datapoint individually
    :type pca_condition: Integer, optional
    :param pca_components: Number of components of the pca
    :type pca_components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :return:
    """
    if pca_type == 1:  # flatten the channel*time dimension and apply pca
        data, pca = reshape_pca(data, pca_components, fit_pca)
    elif pca_type == 2:  # reduce the numbers of channels with pca
        data, pca = channel_pca(data, pca_components, pca_condition, fit_pca)
    elif pca_type == 3:  # reduce the time component with pca
        data, pca = time_pca(data, pca_components, pca_condition, fit_pca)

    return np.array(data), pca


def reshape_pca(data, components, fit_pca=None):
    """ Applies PCA on the flattened inout data
    :param data: eeg data
    :type data: array of shape trials, channel, time
    :param components: number of components of the pca
    :type components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :return pca transformed data and the pca object
    :rtype array of shape trials, channels*time and the fit pca object """
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    if components:
        try:
            pca = PCA(n_components=components)
            pca.fit(data)
            t_data = pca.transform(data)
            return t_data, pca
        except ValueError:
            print(f'Please choose a pca component number smaller than {data.shape[0]}. '
                  f'You do not have enough samples for the component number.')
    if fit_pca:
        t_data = fit_pca.transform(data)
        return t_data, fit_pca


def channel_pca(data, components, pca_condition=0, fit_pca=None):
    """ Applies PCA on the channel part of the input data
    :param data: eeg data
    :type data: array of shape trials, channel, time
    :param components: number of components of the pca
    :type components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :param pca_condition: determines if the pca is fit on the global mean or each trial individually
    :type pca_condition: Integer, 0: global mean, 1: individually
    :return pca transformed data and the pca object
    :rtype array of shape trials, channels, time and the fit pca object """
    if components <= 1:
        pca_condition = 0
    t_data = []
    pca_ = PCA(n_components=int(components))
    if components:
        if pca_condition == 0:
            mean_data = np.mean(data, axis=0)
            pca_.fit(mean_data.T)
            t_data = [pca_.transform(elem.T).T for elem in data]

        else:

            for elem in data:
                pca_ = PCA(n_components=int(components))
                pca_.fit(elem.T)
                temp = pca_.transform(elem.T)
                t_data.append(temp.T)

    if fit_pca != None:
        if pca_condition == 0:
            t_data = [fit_pca.transform(elem.T).T for elem in data]
        else:

            for elem in data:
                temp = fit_pca.transform(elem.T)
                t_data.append(temp.T)
            data = t_data
        pca_ = fit_pca

    return t_data, pca_


def time_pca(data, components, pca_condition=0, fit_pca=None):
    """ Applies PCA on the time dimension of the input data
    :param data: eeg data
    :type data: array of shape trials, channel, time
    :param components: number of components of the pca
    :type components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :param pca_condition: determines if the pca is fit on the global mean or each trial individually
    :type pca_condition: Integer, 0: global mean, 1: individually
    :return pca transformed data and the pca object
    :rtype array of shape trials, channels, time and the fit pca object """
    if components <= 1:
        pca_condition = 0

    t_data = []
    pca_ = PCA(n_components=int(components))
    if components:
        if pca_condition == 0:
            mean_data = np.mean(data, axis=0)
            pca_.fit(mean_data)
            t_data = [pca_.transform(elem) for elem in data]

        else:

            for elem in data:
                pca_ = PCA(n_components=int(components))
                pca_.fit(elem)
                temp = pca_.transform(elem)
                t_data.append(temp)

        if fit_pca:
            if pca_condition == 0:
                t_data = [fit_pca.transform(elem) for elem in data]

            else:
                for elem in data:
                    t_data.append(fit_pca.transform(elem))

            pca_ = fit_pca
    return t_data, pca_


""" Training Methods"""


def k_fold_training(data, model, loss='categorical_crossentropy', optimizer='adam',
                    epochs=10, batch_size=10, folds=10, random_state=None, verbose=0,
                    save=False, model_name='model', filename='model',
                    pca_type=2, pca_components=46, pca_condition=0):
    """ Training the model k-folds times on k-fold splits of the data

    :param data: data consisting of eeg-data and event labels
    :type data: tuple of arrays
    :param model: a tensorflow model that can be trained
    :type model: tensorflow Model
    :param loss: loss function for the training model
    :type loss: keras loss function
    :param optimizer: optimizer for the training model
    :type optimizer: keras optimizer
    :param epochs: number of epochs used to train
    :type epochs: integer
    :param batch_size: size of the batches used to batch the data
    :type batch_size: integer
    :param folds: number of folds created from the data to train on
    :type folds: integer
    :param random_state: if given, a number that keeps the random state fixed for replicability
    :type random_state: integer
    :param verbose: if 0 no training progress is printed, if 1 or 2 training progress is printed
    :type verbose: integer
    :param save: if true, all trained models will be saved
    :type save: Boolean
    :param model_name: name of the model, used when saving the models
    :type model_name: String
    :param filename: name of the results file
    :type filename: String
    :param pca_type: type of pca applied to the pretraining data
    :type pca_type: Integer between 0 an 2
    :param pca_components: number of components used while applying the pca
    :type pca_components: Integer >0 or a Float between 0 and 1
    :param pca_condition: condition of the pca type
    :type pca_condition: integer either 0 or 1
    :return: hist: the training history, eval: the evaluation history, cvs: cumulative variance scores,
    path: the path to where the model is saved
    :rtype: hist: array of dictionaries, eval: array, cvs: array, path: string
    """
    # split data into eeg-data and events
    eeg, events = data

    # empty lists to save training and testing results
    hist = []
    eval_scores = []
    cvscores = []
    path = ''

    # Split-counter
    i = 1
    # K-Fold Training
    for train_index, test_index in StratifiedKFold(folds, shuffle=True, random_state=random_state).split(eeg, events):
        tf.keras.backend.clear_session()
        print('START SPLIT', i)
        # create training and test data  from the split indexes
        train_dataset, pca = data_preprocessing(eeg[train_index], events[train_index], pca_type=pca_type,
                                                pca_components=pca_components, pca_condition=pca_condition,
                                                batch_size=batch_size)
        test_dataset, _ = data_preprocessing(eeg[test_index], events[test_index], pca_type=pca_type,
                                             pca_components=pca_components, pca_condition=pca_condition,
                                             batch_size=batch_size, fit_pca=pca)

        # Initialize model
        model = model
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # Fit model and save results
        temp = model.fit(train_dataset, epochs=epochs, verbose=verbose,
                         validation_data=test_dataset)
        hist.append(temp.history)

        # Evaluate model and save results
        score = model.evaluate(test_dataset, verbose=0)
        eval_scores.append(score)
        cvscores.append(score[1] * 100)
        # give the cumulative accuracy
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        # save the current model
        if save:
            path = f"test_models/Training/{model_name}_{i}_e{epochs}_bs{batch_size}_cvacc{int(score[1] * 100)}"
            tf.keras.models.save_model(model, path)

        # elevate Split-counter
        i += 1

    # print the results of the current fold to allow easy tracking of progress
    print("Validation acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    # save results in a txt file for later use
    f = open(f"results/{filename}_e{epochs}_f{folds}_bs{batch_size}_results.txt", "a")
    f.write('\n'.join([f"Average Accuracy: {np.mean(cvscores)}", f"STD: {np.std(cvscores)}",
                       f'{eval_scores}',
                       f'Model history: {hist}', "\n\n\n"]))
    f.close()

    return hist, eval_scores, cvscores, path


def pretraining(data, model, loss='categorical_crossentropy', optimizer='adam',
                epochs=10, batch_size=10, train_size=0.9,
                save=False, model_name='model', filename='model',
                pca_type=2, pca_components=46, pca_condition=0):
    """ Pretraining a model on a train-test split of the given data

    :param data: data consisting of eeg-data and event labels
    :type data: tuple of arrays
    :param model: a tensorflow model that can be trained
    :type model: tensorflow Model
    :param loss: loss function for the training model
    :type loss: keras loss function
    :param optimizer: optimizer for the training model
    :type optimizer: keras optimizer
    :param epochs: number of epochs used to train
    :type epochs: integer
    :param batch_size: size of the batches used to batch the data
    :type batch_size: integer
    :param train_size: Percentage of the data used for testing
    :type train_size: float between 0 and 1
    :param save: indicates if model should be saved
    :type save: boolean
    :param model_name: name of the model used when saved
    :type model_name String
    :param filename: name of the txt file the training and testing results are stored in
    :type filename: String
    :param pca_type:type of pca applied to the pretraining data
    :type pca_type: Integer between 0 an 2
    :param pca_components: number of components used while applying the pca
    :type pca_components: Integer >0 or a Float between 0 and 1
    :param pca_condition: condition of the pca type
    :type pca_condition: integer either 0 or 1
    :return: hist: the training history, eval: the evaluation history, cvs: cumulative variance scores,
    path: the path to where the model is saved
    :rtype: hist: array of dictionaries, eval: array, cvs: array, path: string
    """
    # split input data into eeg data and events
    eeg, events = data

    # params to save training hystory
    hist = []
    path = ''

    # initialize model with loss and optimizer
    model = model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # training loop
    for train_index, test_index in StratifiedShuffleSplit(1, test_size=1 - train_size,
                                                          train_size=train_size, random_state=1).split(eeg, events):
        tf.keras.backend.clear_session()
        # create training and test data
        train_dataset, pca = data_preprocessing(eeg[train_index], events[train_index], pca_type=pca_type,
                                                pca_components=pca_components, pca_condition=pca_condition,
                                                batch_size=batch_size)
        test_dataset, _ = data_preprocessing(eeg[test_index], events[test_index], pca_type=pca_type,
                                             pca_components=pca_components, pca_condition=pca_condition,
                                             batch_size=batch_size, fit_pca=pca)

        # Fit model and save training history
        temp = model.fit(train_dataset, epochs=epochs, verbose=0, validation_data=test_dataset)
        hist.append(temp.history)

        # Evaluate model
        score = model.evaluate(test_dataset, verbose=0)
    # System output to track model accuracy
    print("Validation acc: %.2f%%" % (score[1] * 100))

    # save model
    if save:
        folder = f"test_models/Pretraining/"
        name = f"{model_name}_e{epochs}_bs{batch_size}_vacc{int((score[1] * 100))}"
        path = folder + name
        tf.keras.models.save_model(model, path)

    # write a results file for later
    f = open(f"results/{filename}_e{epochs}_bs{batch_size}_results.txt", "a")
    f.write('\n'.join([f"Accuracy: {score[1]}",
                       f'{score}', f'Model history: {hist}', "\n\n\n"]))
    f.close()

    return hist, [score], path


def ff_training(data, model, loss='categorical_crossentropy', optimizer='adam',
                epochs=10, batch_size=10, verbose=0,
                save=False, model_name='model', filename='model',
                pca_type=1, pca_components=46, pca_condition=0):
    """ Training the model k-folds times on k-fold splits of the data

    :param data: data consisting of eeg-data and event labels
    :type data: tuple of arrays
    :param model: a tensorflow model that can be trained
    :type model: tensorflow Model
    :param loss: loss function for the training model
    :type loss: keras loss function
    :param optimizer: optimizer for the training model
    :type optimizer: keras optimizer
    :param epochs: number of epochs used to train
    :type epochs: integer
    :param batch_size: size of the batches used to batch the data
    :type batch_size: integer
    :param verbose: if 0 no training progress is printed, if 1 or 2 training progress is printed
    :type verbose: integer
    :param save: if true, all trained models will be saved
    :type save: Boolean
    :param model_name: name of the model, used when saving the models
    :type model_name: String
    :param filename: name of the results file
    :type filename: String
    :param pca_type: type of pca applied to the pretraining data
    :type pca_type: Integer between 0 an 2
    :param pca_components: number of components used while applying the pca
    :type pca_components: Integer >0 or a Float between 0 and 1
    :param pca_condition: condition of the pca type
    :type pca_condition: integer either 0 or 1
    :return: hist: the training history, eval: the evaluation history, cvs: cumulative variance scores,
    path: the path to where the model is saved
    :rtype: hist: array of dictionaries, eval: array, cvs: array, path: string
    """
    # split data into eeg-data and events
    eeg, events = data

    # empty lists to save training and testing results
    hist = []
    eval_scores = []
    cvscores = []
    path = ''

    x, y = data_preprocessing(eeg, events, pca_type, pca_components,
                              pca_condition, batch_size, batched=False)

    tf.keras.backend.clear_session()

    # Initialize model
    model = model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # Fit model and save results
    temp = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                     validation_split=0.1)
    hist.append(temp.history)

    score = np.mean(temp.history['val_accuracy']) * 100
    eval_scores.append([np.mean(temp.history['val_loss']), np.mean(temp.history['val_accuracy'])])
    cvscores.append(score)

    # save the current model
    if save:
        path = f"test_models/Training/{model_name}_e{epochs}_bs{batch_size}_cvacc{score}"
        tf.keras.models.save_model(model, path)

    # print the results of the current fold to allow easy tracking of progress
    print("Validation acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    # save the training results in a txt file for later use
    f = open(f"results/{filename}_e{epochs}_bs{batch_size}_results.txt", "a")
    f.write('\n'.join([f"Average Accuracy: {np.mean(cvscores)}", f"STD: {np.std(cvscores)}",
                       f'{eval_scores}',
                       f'Model history: {hist}', "\n\n\n"]))
    f.close()

    return hist, eval_scores, cvscores, path
