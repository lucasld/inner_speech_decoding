import mne
import numpy as np
import tensorflow as tf

import utilities


def load_data(subjects=range(1,11), path='./dataset'):
    """Load EEG-Data and Event-Data into a numpy array.
    
    :param subjects: array of subjects from which the sessions should be
        loaded, defaults to subjects 1 to 10
    :type subjects: array of integers, optional
    :param path: path to dataset-directory, defaults to './dataset'
    :type path: string, optional
    :return: eeg-data and event data in seperate arrays
    :rtype: tuple of two numpy arrays
    """
    data_collection = np.empty((0, 128, 1153))
    events_collection = np.empty((0, 4), dtype=int)
    for sub in subjects:
        for ses in [1, 2, 3]:
            utilities.progress_bar(((sub-1)*3+ses)/(3*len(subjects)))
            try:
                path_part = path + '/derivatives/sub-' + str(sub).zfill(2) +\
                            '/ses-0'+ str(ses) + '/sub-' + str(sub).zfill(2) +\
                            '_ses-0' + str(ses)
                # load data
                file_name = path_part + '_eeg-epo.fif'
                data = mne.read_epochs(file_name, verbose='WARNING')
                data_collection = np.append(data_collection, data._data, axis=0)

                # load events
                file_name =  path_part + '_events.dat'
                events = np.load(file_name, allow_pickle=True)
                events_collection = np.append(events_collection, events, axis=0)
            except:
                pass
    
    return data_collection, events_collection


def choose_condition(data, events, condition):
    """Filters out a specific condition from the data.

    :param data: data from which to extract conditions from
    :type data: numpy array
    :param events: event data of shape [trials x 4]
    :type events: numpy array
    :param condition: Condition to be filtered out of the data.
        Conditions that exist are:
        0 <=> Pronounced Speech
        1 <=> Inner Speech
        2 <=> Visualized Condition
    :type condition: string of integer specifying the condition
    :return: eeg data and event data
    :rtype: tuple of two numpy arrays
    """
    # convert condition to the right format
    if type(condition)==str:
        condition = sorted(condition.replace(' ', '').lower())
        if condition == sorted('pronouncedspeech'):
            condition = 0
        elif condition == sorted('innerspeech'):
            condition = 1
        elif condition == sorted('visualizedcondition'):
            condition = 2
        else:
            raise ValueError("The condition-string you provided is wrong!")
    # filter out every sample with the right condition
    keep_pos = events[:, 2] == condition
    data = data[keep_pos]
    events = events[keep_pos]
    return data, events


def preprocessing_pipeline(data, functions=None, args=None, batch_size=32):
    """Apply preproccesing pipeline to the given dataset.
    
    :param data: data to be preprocessed
    :type data: tensorflow 'Dataset'
    :param batch_size: number of elements per batch, defaults to 32
    :type batch_size: integer, optional
    :param functions: functions that should be mapped to the data,
        defaults to None
    :type functions: single function or list of functions, functions take and
        return input and target, optional
    :param args: list of arguments for each of the provided functions,
        defaults to None
    :type args: list of arguments of only one function was provided,
        list of lists of arguments if several functions where provided,
        optional
    :return: preprocessed dataset
    :rtype: tensorflow 'Dataset'
    """
    # map functions provided as arguments to data
    if type(functions) not in [list, tuple] and functions:
        functions = [functions]
        args = [args]
    for func, arg in zip(functions, args) if functions else []:
        data = data.map(lambda input, target: func((input, target), *arg))
    # cache the dataset
    data = data.cache()
    # shuffle, batch and prefetch the dataset
    data = data.shuffle(1000)
    data = data.batch(batch_size)
    data = data.prefetch(100)
    return data




def filter_interval(sample, interval, data_frequency, apply_indices=[0]):
    """Cut out a specific interval from a sample of EEG-Data.
    
    :param sample: sample consisitng of channel_number * data_points
    :type sample: tensor
    :param interval: two values specifying the starting and end point in
        seconds of the interval to be cut out.
        Each sample consists of a 4.5 second EEG-data window.
        These 4.5 seconds consist of:
        Concentration Interval - 0.5 s
        Cue Interval - 0.5 s
        Action Interval - 2.5 s
        Relac Interval - 1 s
    :type interval: list of two floating point numbers
    :param data_frequency: specifies the frequency the provided data was
        measured at
    :type data_frequency: floating point number
    :param apply_indices: specifies on what elemtents of sample to apply the
        function, defaults to [0]
    :type apply_indices: list of integers, optional
    :return: cut sample
    :rtype: tensor
    """
    sample = list(sample)
    start_index_interval = int(interval[0] * data_frequency)
    end_index_interval = int(interval[1] * data_frequency)
    for index in apply_indices:
        sample[index] = sample[index][:, start_index_interval:end_index_interval]
    return sample


def normalization(data, axis=2, epsilon=1e-8):
    """Normalize numpy data.  # z-score standartization

    :param data: dataset of eeg-data
    :type data: numpy array of dimension [samples x channels x N]
    :param axis: axis along which to normalize the data
    :type axis: int or tuple of ints
    :param epsilon: epsilon to prevent division by zero
    :type epsilon: float
    :return: normalized dataset
    :rtype: numpy array
    """
    mean, variance = np.mean(data, axis=axis), np.var(data, axis=axis)
    data_normed = (data - mean[:, :, np.newaxis]) / np.sqrt(variance + epsilon)[:, :, np.newaxis]
    return data_normed


def split_dataset(dataset, splits={'train': 0.7,
                                   'test': 0.15,
                                   'valid': 0.15}):
    """Split a tensorflow dataset into n subsets.

    :param dataset: dataset to be split up
    :type dataset: tf.data.Dataset
    :param splits: specifies the proportion for each subset, defaults to:
        'train' => 70%
        'test' => 15%
        'valid' => 15%
    :type splits: dictonary, values of type float which sum up to max 1,
        optional
    :return: new smaller datasets
    :rtype: dictonary, keys are the same as in the splits-argument and the
        values are tf.data.Dataset's
    """
    assert sum(splits.values()) <= 1, "split-proportions sum to more than 1!"
    datasets = {}
    batch_number = len(list(dataset))
    # iterate through splits to create dataset-subsets
    take = 0
    for i, (key, proportion) in enumerate(splits.items()):
        if i == 0:
            take = int(batch_number * proportion)
            datasets[key] = dataset.take(take)
        else:
            datasets[key] = dataset.skip(take)
            datasets[key] = datasets[key].take(int(batch_number * proportion))
            take += int(batch_number * proportion)
    return datasets