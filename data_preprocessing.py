import mne
import numpy as np
import utilities

def load_data(subjects=range(1,11), path='./dataset'):
    """Load EEG-Data and Event-Data into a numpy array.
    
    :param subjects: array of subjects from which the sessions should be loaded
    :type subjects: array of integers
    :param path: path to dataset-directory
    :type path: string
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

    ...
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