import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA
import data_preprocessing as dp

""" Methods for Plotting PCA data """


def side_by_side(original, reconstructed, indexes):
    """Plot original data and from PCA reconstructed data side by side.

    :param original: array of the original data in the shape (trial, channels, time).
    :type original: array of integers
    :param reconstructed: array of the data reconstructed from the pca in the shape (trial, channels, time).
    :type reconstructed: array of integers
    :param indexes: number of plotted samples
    :type indexes: integer
    :return: image showing the original data on the left and the reconstructed data on the right
    :rtype: pyplot figure
    """
    org = original[indexes]
    rec = reconstructed[indexes]
    middle = np.zeros((org.shape[0], 20))
    pair = np.concatenate((org, middle), axis=1)
    pair = np.concatenate((pair, rec), axis=1)
    plt.figure(figsize=(16, 14))
    plt.imshow(pair, cmap='viridis')
    plt.title('Before and after pca')

    plt.show()


def pca_pic(pca_data, index, dim=1):
    """Plot the PCA of one datapoint as image.

    :param pca_data: array of the pca_transformed data
    :type pca_data: array of integers
    :param index: number of plotted samples
    :type index: integer
    :param dim: dimension of a pca_sample
    :type dim: integer
    :return: image showing the pca transformed data as image
    :rtype: pyplot figure
    """
    plt.figure(figsize=(8, 6))
    if dim != 1:
        plt.imshow(pca_data[index])
    else:
        plt.plot(pca_data[index])
    plt.title('PCA data')
    plt.show()


def difference(original, reconstructed, indexes):
    """Plot the difference between the original data and the from PCA reconstructed data.

    :param original: array of the original data in the shape (trial, channels, time).
    :type original: array of integers
    :param reconstructed: array of the data reconstructed from the pca in the shape (trial, channels, time).
    :type reconstructed: array of integers
    :param indexes: number of plotted samples
    :type indexes: integer
    :return: image showing the difference between the original data and the reconstructed data
    :rtype: pyplot figure
    """
    org = original[indexes]
    rec = reconstructed[indexes]
    diff = org - rec
    plt.figure(figsize=(8, 7))
    plt.imshow(diff)
    plt.title('Difference: Before- after')
    plt.show()


def kFoldVisualization(history, evaluation, epochs = 50, folds=10, batchsize=10, save=False, name="kFoldVisualization"):
    fig, axs = plt.subplots(folds, 2, figsize=(15, 15))

    for i in range(folds):
        temp = history[i]
        axs[i][0].plot(temp['accuracy'])
        axs[i][0].axhline(evaluation[i][1], color='r')
        axs[i][0].axhline(np.average([x[1] for x in evaluation]), color='g', linestyle='--')
        axs[i][0].set(title='Model ' + str(i + 1), ylabel='accuracy', xlabel='epoch')
        axs[i][0].legend(['train', 'val', 'average val'], loc='upper left')
        axs[i][1].plot(temp['loss'])
        axs[i][1].axhline(evaluation[i][0], color='r')
        axs[i][1].axhline(np.average([x[0] for x in evaluation]), color='g', linestyle='--')
        axs[i][1].set(title='Model ' + str(i + 1), ylabel='loss', xlabel='epoch')
        axs[i][1].legend(['train', 'val', 'average val'], loc='upper left')
    if save == True:
        plt.savefig(f'figures/KFold_Cross_Validation/{name}_e{epochs}_bs{batchsize}_f{folds}')
        plt.clf()
    else:
        plt.show()


""" Data wrangling methods"""


def data_prep(subjects=range(1, 11),
              condition=['inner speech', 'pronounced speech', 'visualized condition'],
              time=[1, 3.5]):
    """ Create the initial pca data to train a network
    :param subjects: Number of subjects whomse data will be prepared
    :type subjects: range of integer numbers from 1-11, standard is all subjects
    :param condition: Conditions used of the whole experiment.
    :type condition: list of the condition names, standard is all three conditions
    :param time: Time frame extracted from the whole eeg trial
    :type time: array of floats [start, end]
    :param pca_type: type of pca we want to apply to the data coded as integers.
                        0: no pca
                        1: pca on reshaped data
                        2: pca on the channels
                        3: pca on the timepoints
    :type pca_type: integer, optional
    :param pca_condi: additional parameter only used for pca_type 2 and 3.
                        0: pca is fit on the mean of all datapoints
                        1: pca is fit onto each datapoint inividually
    :type pca_condi: Integer, optional
    :param pca_components: Number of components of the pca
    :type pca_components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :return dictionary of datapoints, events, scaler and pca with the experimental conditions as keys
    :rtype float, float, RobustScaler, PCA
    """
    data, events = dp.load_data(subjects=subjects)
    d = {}
    e = {}
    # oneHot = OneHotEncoder(categories=4)
    # temp = np.array(events[:, 1]).reshape(-1,1)
    # oneHot.fit(temp)
    for condi in condition:
        # print('*** Data Prep: ', condi)
        # print('---Load Data---')
        s = RobustScaler()
        temp_data, temp_events = dp.choose_condition(data, events, condi)
        temp_events = np.array(temp_events[:, 1])  # .reshape(-1,1)
        # print('data shape: ', temp_data.shape)
        # print('events shape: ', temp_events.shape)
        # temp_data = oneHot.transform(temp_events)
        temp_data = temp_data[:, :, int(time[0] * 256):int(time[1] * 256)]
        temp_data = s.fit_transform(temp_data.reshape(-1, temp_data.shape[-1])).reshape(temp_data.shape)
        # print('--- preprocess Data ---')
        # print('data shape: ', temp_data.shape)
        # print('events shape: ', temp_events.shape)
        e[condi] = temp_events
        d[condi] = temp_data
    return d, e


""" Methods to apply PCA on EEG data"""


def apply_pca(data, pca_type=0, pca_condi=0, pca_components=0.98):
    #print('---Apply PCA---')
    if pca_type == 1:
        temp_data = reshape_pca(data, pca_components)
    elif pca_type == 2:
        temp_data = channel_pca(data, pca_components, pca_condi)
    elif pca_type == 3:
        temp_data = time_pca(data, pca_components, pca_condi)

    return np.array(temp_data)


def reshape_pca(input, components):
    """ Applies PCA on the flattened inout data
    :param input: eeg data
    :type input: array of shape trials, channel, time
    :param components: number of components of the pca
    :type components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :return pca transformed data and the pca object
    :rtype array of shape trials, channels*time and the fit pca object """
    data = input.reshape(input.shape[0], input.shape[1] * input.shape[2])
    pca = PCA(n_components=components)
    data = pca.fit_transform(data)
    return data


def channel_pca(data, components, pca_condi=0):
    """ Applies PCA on the channel part of the input data
    :param input: eeg data
    :type input: array of shape trials, channel, time
    :param components: number of components of the pca
    :type components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :param pca_condi: determines if the pca is fit on the global mean or each trial individually
    :type pca_condi: Integer, 0: global mean, 1: individually
    :return pca transformed data and the pca object
    :rtype array of shape trials, channels, time and the fit pca object """
    if components <= 1:
        pca_condi = 0

    if pca_condi == 0:
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


def time_pca(data, components, pca_condi=0):
    """ Applies PCA on the time dimension of the input data
    :param data: eeg data
    :type data: array of shape trials, channel, time
    :param components: number of components of the pca
    :type components: Either float < 1 to tell the percentage of variance explained
                        or an Integer defining the total number of components
    :param pca_condi: determines if the pca is fit on the global mean or each trial individually
    :type pca_condi: Integer, 0: global mean, 1: individually
    :return pca transformed data and the pca object
    :rtype array of shape trials, channels, time and the fit pca object """
    if components <= 1:
        pca_condi = 0

    if pca_condi == 0:
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

def data_preprocessing(data, events, batchsize = 10):
    data = tf.data.Dataset.from_tensor_slices(data, events)

    #expand dimension
    data = data.map(lambda image, event: (tf.expand_dims(image, axis = -1), event))
    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()
    #shuffle, batch, prefetch
    data = data.shuffle(100)
    data = data.batch(batchsize)
    data = data.prefetch(batchsize*2)
    #return preprocessed dataset
    return data
def kFoldTraining(data, model, loss='sparse_categorical_crossentropy', optimizer='adam',
                  epochs=10, batchsize=10, folds=10,
                  save=False, model_name='model', filename='model',
                  pca_type=2, pca_components=46, pca_condition=0):
    X, Y = data
    hist = []
    eval = []
    cvscores = []

    i = 1

    for train_index, test_index in StratifiedKFold(folds, shuffle=True, random_state=5).split(X, Y):
        tf.keras.backend.clear_session()
        print('START SPLIT', i)
        # print('--- create training and test data ---')
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # print('--- PCA ---')

        x_train = apply_pca(x_train, pca_type=pca_type, pca_components=pca_components, pca_condi=pca_condition)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = apply_pca(x_test, pca_type=pca_type, pca_components=pca_components, pca_condi=pca_condition)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        print('--- Initialize model ---')
        model = model
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # print('--- Fit model ---')
        temp = model.fit(x_train, y_train, batch_size=batchsize, epochs=epochs, verbose=0, validation_data=(x_test, y_test))

        hist.append(temp.history)

        # print('--- Evaluate model ---')
        score = model.evaluate(x_test, y_test, verbose=0)
        eval.append(score)
        cvscores.append(score[1] * 100)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
        if save:
            path = f"test_models/{model_name}_e{epochs}_bs{batchsize}_cv{i}"
            tf.keras.models.save_model(model, path)

        i += 1
    print("Validation acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    f = open(f"results/{filename}_e{epochs}_f{folds}_bs{batchsize}_results.txt", "a")
    f.write('\n'.join([f"Average Accuracy: {np.mean(cvscores)}", f'{eval}', f'Model history: {hist}', "\n\n\n"]))
    f.close()

    return hist, eval, cvscores, path



def pretraining(data, model, loss='sparse_categorical_crossentropy', optimizer='adam',
                  epochs=10, batchsize=10, folds=10,
                  save=False, model_name='model', filename='model',
                  pca_type=2, pca_components=46, pca_condition=0):
    X, Y = data
    hist = []
    eval = []
    cvscores = []

    i = 1

    model = model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    np.random.shuffle()
    for train_index, test_index in StratifiedShuffleSplit(1, test_size=0.1, train_size=0.9, random_state=1).split(X, Y):
        tf.keras.backend.clear_session()
        print('START SPLIT', i)
        # print('--- create training and test data ---')
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # print('--- PCA ---')

        x_train = apply_pca(x_train, pca_type=pca_type, pca_components=pca_components, pca_condi=pca_condition)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = apply_pca(x_test, pca_type=pca_type, pca_components=pca_components, pca_condi=pca_condition)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


        # print('--- Fit model ---')
        temp = model.fit(x_train, y_train, batch_size=batchsize, epochs=epochs, verbose=0, validation_data=(x_test, y_test))

        hist.append(temp.history)

        # print('--- Evaluate model ---')
        score = model.evaluate(x_test, y_test, verbose=0)
        eval.append(score)
        cvscores.append(score[1] * 100)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        i += 1
    print("Validation acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    if save:
        folder = f"test_models/Pretraining/"
        name = f"{model_name}_e{epochs*folds}_bs{batchsize}_vacc{int(np.mean(cvscores))}_vstd{int(np.std(cvscores))}"
        path = folder + name
        tf.keras.models.save_model(model, path)

    f = open(f"results/{filename}_e{epochs}_f{folds}_bs{batchsize}_results.txt", "a")
    f.write('\n'.join([f"Average Accuracy: {np.mean(cvscores)}", f'{eval}', f'Model history: {hist}', "\n\n\n"]))
    f.close()

    return hist, eval, cvscores, path
