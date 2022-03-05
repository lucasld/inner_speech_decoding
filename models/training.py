import tensorflow as tf
import utilities
import numpy as np


def train_step(model, input, target, loss_function, optimizer, acc_type):
    """Apply optimizer to all trainable variables of a model to
    minimize the loss (loss_function) between the target output and the
    predicted ouptut.
    :param model: model to train
    :type mdoel: tensorflow model
    :param input: input to the model
    :type input: tf.Tensor
    :param target: target output with repect to the input
    :type target: tf.Tensor
    :param loss_function: loss function used to calculate loss of the model
    :type loss_function: function from the tf.keras.losses module
    :param optimizer: optimizer used to apply gradients to the models
        trainable variables
    :type optimizer: function from the tf.keras.optimizers module
    :return: the loss and the accuracy of the models prediction
    :rtype: tuple of two floats
    """
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    # apply gradients to the trainable variables using a optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = accuracy(prediction, target, acc_type)
    return loss, acc


def test(model, test_data, loss_function, acc_type):
    """Calculate the mean loss and accuracy of the model over all elements
    of test_data.
    :param model: model to train
    :type mdoel: tensorflow model
    :param test_data: model is evaulated for test_data
    :type test_data: tensorflow 'Dataset'
    :param loss_function: loss function used to calculate loss of the model
    :type loss_function: function from the tf.keras.losses module
    :return: mean loss and mean accuracy for all datapoints
    :rtype: tuple of two floats
    """
    # aggregator lists for tracking the loss and accuracy
    test_accuracy_agg = []
    test_loss_agg = []
    # iterate over all input-target pairs in test_data
    for (input, target) in test_data:
        prediction = model(input)
        loss = loss_function(target, prediction)
        acc = accuracy(prediction, target, acc_type)
        # add loss and accuracy to aggregators
        test_loss_agg.append(loss.numpy())
        test_accuracy_agg.append(np.mean(acc))
    # calculate mean loss and accuracy
    test_loss = tf.reduce_mean(test_loss_agg)
    test_accuracy = tf.reduce_mean(test_accuracy_agg)
    return test_loss, test_accuracy


def accuracy(pred, target, type='arg_max'):
    """Calucalte accuracy between a prediction and a target.
    
    :param pred: a prediction that the model made
    :type pred: tf.Tensor of floats
    :param target: target that model should have predicted
    :type target: tf.Tensor of floats
    """
    if type == 'arg_max':
        same_prediction = np.argmax(target, axis=1) == np.argmax(pred, axis=1)
        acc = np.mean(same_prediction)
    elif type == 'mse':
        target = tf.cast(target, tf.float32)
        acc = tf.math.reduce_mean(tf.math.square(tf.math.subtract(target, pred)))
    return acc


def training(input_model, datasets, loss_function, optimizer, acc_type, epochs=10):
    """Training a model on a dataset for a certain amount of epochs.
    :param input_model: model to be trained
    :type input_model: model of type CustomModel
    :param datasets: train, validation and test datasets
    :type datasets: dictionary containing tf datasets keyed with: 'train',
        'test', 'valid'
    :param loss_function: loss function used to calculate loss of the model
    :type loss_function: function from the tf.keras.losses module
    :param optimizer: optimizer used to apply gradients to the models
        trainable variables
    :type optimizer: function from the tf.keras.optimizers module
    :param epochs: number of epochs to train on the dataset
    :type epochs: integer
    :return: losses and accuracies
    :rtype: tuple containing two dictonaries for the losses and accuracies.
        These are keyed like the dataset with 'train', 'valid' and 'test'
    """
    tf.keras.backend.clear_session()
    # Initialize lists for tracking loss and accuracy
    losses = {'train':[], 'valid':[], 'test':0}
    accuracies = {'train':[], 'valid':[], 'test':0}
        
    # Train-Dataset
    train_loss, train_accuracy = test(input_model, datasets['train'], loss_function, acc_type)

    losses['train'].append(train_loss)
    accuracies['train'].append(train_accuracy)

    valid_loss, valid_accuracy = test(input_model, datasets['valid'], loss_function, acc_type)
    #valid_losses.append(valid_loss)
    #valid_accuracies.append(valid_accuracy)
    losses['valid'].append(valid_loss)
    accuracies['valid'].append(valid_accuracy)
    # grapher
    graphs = utilities.TrainingGrapher(2, 1, name="Training Progress", supxlabel='Epochs', x_scale=[[[0, epochs], [0, epochs]]])
    # Training for epochs
    for epoch in range(1, epochs+1):
        last_valid_acc = np.round(accuracies['valid'][-1], 3)
        graphs.update([range(epoch) for _ in range(2)], [accuracies['train'], losses['train']])
        #print(f"Epoch {str(epoch)} starting with validation accuracy of {last_valid_acc} and validation loss of {losses['valid'][-1]:.4f}")
        epoch_loss_agg = []
        epoch_accuracy_agg = []
        for input, target in datasets['train']:
            train_loss, train_accuracy = train_step(
                input_model, input, target, loss_function, optimizer, acc_type
            )
            epoch_loss_agg.append(train_loss)
            epoch_accuracy_agg.append(train_accuracy)
        # track training loss and accuracy
        losses['train'].append(tf.reduce_mean(epoch_loss_agg))
        accuracies['train'].append(tf.reduce_mean(epoch_accuracy_agg))
        # track loss and accuracy for test-dataset
        valid_loss, valid_accuracy = test(input_model, datasets['valid'], loss_function, acc_type)
        losses['valid'].append(valid_loss)
        accuracies['valid'].append(valid_accuracy)
    test_loss, test_accuracy = test(input_model, datasets['test'], loss_function, acc_type)

    losses['test'] = test_loss
    accuracies['test'] = test_accuracy    

    return losses, accuracies


class Trainer:
    def __init__(self, model, datasets,
                 optimizer, loss_function, accuracy_function=None):
        self.model = model
        self.datasets = datasets
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function
        # Metrics
        self.losses = {'train':[], 'test':[], 'valid':[]}
        self.accuracies = {'test':[], 'valid':[]}

    def train_epoch(self):
        # lists that aggregate the loss and accuracy for each batch
        loss_agg = []
        for input_batch, target_batch in self.datasets['train']:
            batch_loss = self.train_step(
                input_batch,
                target_batch
            )
            loss_agg.append(batch_loss)
        self.losses['train'].append(float(tf.reduce_mean(loss_agg)))
        # test
        test_loss, test_acc = self.test('test')
        self.losses['test'].append(float(test_loss))
        self.accuracies['test'].append(float(test_acc))


    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            prediction = self.model(X)
            loss = self.loss_function(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # apply gradients to the trainable variables using optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return loss
    
    def test(self, data_key):
        data = self.datasets[data_key]
        loss_agg = []
        acc_agg = []
        for input, target in data:
            output = self.model(input)
            loss_agg.append(self.loss_function(output, target))
            if self.accuracy_function: acc_agg.append(self.accuracy_function(output, target))
        loss = tf.reduce_mean(loss_agg)
        acc = tf.reduce_mean(acc_agg)
        return loss, acc

