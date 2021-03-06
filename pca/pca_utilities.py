import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

""" Methods for Plotting data 
        3 Methods to visualize application and inverse application of pca
            1) side_by_side: shows the data before dimension reduction and the data after the 
                dimension reduction is inverse
            2) pca_visual: visualizes the dimension reduced data
            3) difference: plots the original data minus the inverse pca application data to show what 
                information is lost during dimension reduction
        
        2 Methods to visualize the training, test and validation data
            1) kFoldVisualization: visualizes the data of k-Fold Cross validation training
                - each fold has its own subplot for accuracy and loss
            2) pretrainingVisualization: visualizes the pretraining
                - accuracy and loss hav their own subplots
"""


def side_by_side(original, reconstructed, indexes):
    """Plot original data and from pca reconstructed data side by side.

    :param original: array of the original data in the shape (trial, channels, time)
    :type original: array of integers
    :param reconstructed: array of the data reconstructed from the pca in the shape (trial, channels, time)
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


def pca_visual(pca_data, index, dim=1):
    """Plot the pca of one datapoint as image.
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
    plt.title('pca data')
    plt.show()


def difference(original, reconstructed, indexes):
    """Plot the difference between the original data and from pca reconstructed data.

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


def k_fold_visualization(history, evaluation, batch_size=10, epochs=10,
                         save=False, name="Image", folder='figures/KFold_Cross_Validation'):
    """ Visualize the results of k-Fold training with subplots for accuracy and loss for each fold in a new row.

    :param history: history of the k-fold training
    :type history: array of dictionaries containing the training metrics
                [accuracy, loss, val_accuracy, val_loss]
    :param evaluation: results of the tensorflow function evaluation
    :type evaluation: array
    :param batch_size: batch size used for training used in the figure name
    :type batch_size: integer
    :type epochs: number of epochs
    :param epochs: Integer
    :param save: variable to determine if the image is saved to a file or directly shown
    :type save: boolean
    :param name: name of the image
    :type name: String
    :param folder: name of the folder or path, where the image is supposed to be saved
    :type folder: String
    :return: None
    """
    if len(history) == 1:
        figsize = (15, 8)
    else:
        figsize = (15, 15)
    fig, axs = plt.subplots(len(history), 2, figsize=figsize, squeeze=False)
    for i in range(len(history)):
        temp = history[i]
        axs[i][0].plot(temp['accuracy'])
        axs[i][0].plot(temp['val_accuracy'])
        axs[i][0].axhline(evaluation[i][1], color='r')
        axs[i][0].axhline(np.average([x[1] for x in evaluation[:]]), color='g', linestyle='--')
        axs[i][0].set(title='Model ' + str(i + 1), ylabel='accuracy', xlabel='epoch')
        axs[i][0].legend(['train', 'val', 'test', 'average test'], loc='upper left')
        axs[i][1].plot(temp['loss'])
        axs[i][1].plot(temp['val_loss'])
        axs[i][1].axhline(evaluation[i][0], color='r')
        axs[i][1].axhline(np.average([x[0] for x in evaluation[:]]), color='g', linestyle='--')
        axs[i][1].set(title='Model ' + str(i + 1), ylabel='loss', xlabel='epoch')
        axs[i][1].legend(['train', 'val', 'test', 'average test'], loc='upper left')
    if save:
        plt.savefig(f'{folder}/{name}_e{epochs}_bs{batch_size}_f{len(history)}')
        plt.clf()
    else:
        plt.show()


"""
def training_rate_tester(data, model, loss, optimizer, epochs=10,
                         rate=0.001, counter=10, stepsize = 0.002):
    validation = []
    for i in range(1,counter+1):
        print(f'-> Test training rate: {rate} ')
        optimizer.learning_rate = rate
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        temp = model.fit(data, epochs=epochs, verbose=0)
        rate += stepsize -0.001
        validation.append([rate, np.mean(temp.history['loss']), np.mean(temp.history['accuracy'])])
    return validation
"""
