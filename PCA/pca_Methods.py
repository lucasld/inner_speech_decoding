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


def data_preprocessing(x, y, pca_type, pca_components, pca_condition, batch_size=10):
    # Apply PCA on the eeg data
    if pca_type:
        x = apply_pca(x, pca_type=pca_type, pca_components=pca_components, pca_condition=pca_condition)

    # Prepare the tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices((x, y))

    data = data.map(lambda image, label: (tf.cast(image, tf.float32), tf.one_hot(int(label), 4)))
    data = data.map(lambda image, label: (tf.expand_dims(image, axis=-1), label))

    # cache the dataset
    data = data.cache()
    # shuffle, batch and prefetch the dataset
    data = data.shuffle(100)
    data = data.batch(batch_size)
    data = data.prefetch(10)

    # return preprocessed dataset
    return data


""" Methods to apply PCA on EEG data"""


def apply_pca(data, pca_type=2, pca_condition=0, pca_components=0.98):
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
        data = reshape_pca(data, pca_components)
    elif pca_type == 2:  # reduce the numbers of channels with pca
        data = channel_pca(data, pca_components, pca_condition)
    elif pca_type == 3:  # reduce the time component with pca
        data = time_pca(data, pca_components, pca_condition)

    return np.array(data)


def reshape_pca(data, components):
    """ Applies PCA on the flattened inout data
    :param data: eeg data
    :type data: array of shape trials, channel, time
    :param components: number of components of the pca
    :type components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :return pca transformed data and the pca object
    :rtype array of shape trials, channels*time and the fit pca object """
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    pca = PCA(n_components=components)
    data = pca.fit_transform(data)
    return data


def channel_pca(data, components, pca_condition=0):
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

    if pca_condition == 0:
        pca_ = PCA(n_components=int(components))
        mean_data = np.mean(data, axis=0)
        pca_.fit(mean_data.T)
        data = [pca_.transform(elem.T).T for elem in data]

    else:
        pca_ = []
        data = []
        for elem in data:
            pca = PCA(n_components=int(components))
            temp = pca.fit_transform(elem.T).T
            pca_.append(pca)
            data.append(temp)
    return data


def time_pca(data, components, pca_condition=0):
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

    if pca_condition == 0:
        pca_ = PCA(n_components=int(components))
        mean_data = np.mean(data, axis=0)
        pca_.fit(mean_data)
        data = [pca_.transform(elem) for elem in data]

    else:
        pca_ = []
        data = []
        for elem in data:
            pca = PCA(n_components=int(components))
            temp = pca.fit_transform(elem)
            pca_.append(pca)
            data.append(temp)

    return data


""" Training Methods"""


def k_fold_training(data, model, loss='categorical_crossentropy', optimizer='adam',
                    epochs=10, batch_size=10, folds=10, random_state=None,
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
        train_dataset = data_preprocessing(eeg[train_index], events[train_index], pca_type, pca_components,
                                           pca_condition, batch_size)
        test_dataset = data_preprocessing(eeg[test_index], events[test_index], pca_type, pca_components,
                                          pca_condition, batch_size)

        # Initialize model
        model = model
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # Fit model and save results
        temp = model.fit(train_dataset, epochs=epochs, verbose=0, validation_data=test_dataset)
        hist.append(temp.history)

        # Evaluate model and save results
        score = model.evaluate(test_dataset, verbose=0)
        eval_scores.append(score)
        cvscores.append(score[1] * 100)
        # give the cumulative accuracy
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        # save the current model
        if save:
            path = f"test_models/{model_name}_e{epochs}_bs{batch_size}_cv{i}"
            tf.keras.models.save_model(model, path)

        # elevate Split-counter
        i += 1
    print("Validation acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    f = open(f"results/{filename}_e{epochs}_f{folds}_bs{batch_size}_results.txt", "a")
    f.write('\n'.join([f"Average Accuracy: {np.mean(cvscores)}", f'{eval_scores}', f"STD: {np.std(cvscores)}",
                       f'Model history: {hist}', "\n\n\n"]))
    f.close()

    return hist, eval_scores, cvscores, path


def pretraining(data, model, loss='categorical_crossentropy', optimizer='adam',
                epochs=10, batch_size=10, test_size=0.1, train_size=0.9,
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
    :param test_size: Percentage of the data used for training
    :type test_size: float between 0 an 1
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
    eeg, events = data
    hist = []
    eval_score = []
    scores = []
    path = ''

    model = model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    for train_index, test_index in StratifiedShuffleSplit(1, test_size=test_size,
                                                          train_size=train_size, random_state=1).split(eeg, events):
        tf.keras.backend.clear_session()
        # create training and test data
        train_dataset = data_preprocessing(eeg[train_index], events[train_index], pca_type, pca_components,
                                           pca_condition, batch_size)
        test_dataset = data_preprocessing(eeg[test_index], events[test_index], pca_type, pca_components,
                                          pca_condition, batch_size)
        # Fit model
        temp = model.fit(train_dataset, epochs=epochs, verbose=0, validation_data=test_dataset)

        hist.append(temp.history)

        # Evaluate model
        score = model.evaluate(test_dataset, verbose=0)
        eval_score.append(score)
        scores.append(score[1] * 100)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    print("Validation acc: %.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    if save:
        folder = f"test_models/Pretraining/"
        name = f"{model_name}_e{epochs}_bs{batch_size}_vacc{int(np.mean(scores))}_vstd{int(np.std(scores))}"
        path = folder + name
        tf.keras.models.save_model(model, path)

    f = open(f"results/{filename}_e{epochs}_bs{batch_size}_results.txt", "a")
    f.write('\n'.join([f"Average Accuracy: {np.mean(scores)}", f"STD: {np.std(scores)}",
                       f'{eval_score}', f'Model history: {hist}', "\n\n\n"]))
    f.close()

    return hist, eval_score, scores, path
