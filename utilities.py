import matplotlib.pyplot as plt
import tensorflow as tf


def progress_bar(proportion, size=100):
    """Tool to visualize the progress of any process.
    
    :param proportion: proportion of the task that is done
    :type proportion: floating point number
    :param size: target size of the progress-bar, defaults to 100
    :type size: integer, optional
    """
    elements = ['[','', ']', f'{round(proportion*100)}%']
    bar = ''
    for i in range(size):
        p = i / size
        if p <= proportion:
            elements[1] += f'\033[38;2;{int(255*(1-p))};{int(255*p)};{0}m#'
        else:
            elements[1] += ' '
    print('\033[0m'.join(elements), end='\r')


def plot_eeg_data(sample):
    plt.imshow(sample)