# -*- coding: utf-8 -*-
"""EEG/MEG_5Layer_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZUn4JY1bF1-yUtOatK6vobMHIl8F41LB

**EEG/MEG Convolutional Classifcation Network**

This network is largely modeled after the network used in the paper by Lun et al published 2020 (https://doi.org/10.3389/fnhum.2020.00338)
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Normalization, LeakyReLU, Activation, BatchNormalization, MaxPool2D


class ConvNet2(Model):

    def __init__(self):
        """
        init: constructor of model
        call: performs forward pass of model
        """
        super(ConvNet2, self).__init__()

        self.layer_list = [
            Conv2D(filters=64, kernel_size=(3,3)),
            LeakyReLU(),
            Dropout(0.3),
            Conv2D(filters=128, kernel_size=(3,3)),
            BatchNormalization(),
            LeakyReLU(),
            MaxPool2D(),
            Conv2D(filters=128, kernel_size=(3,3)),
            LeakyReLU(),
            Dropout(0.3),
            MaxPool2D(),
            Conv2D(filters=64, kernel_size=(3,3)),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.3),
            MaxPool2D(),
            Conv2D(filters=32, kernel_size=(3,3)),
            BatchNormalization(),
            LeakyReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(4, activation='softmax')
        ]

    @tf.function
    def call(self, inputs, training=None) -> tf.Tensor:

        """ Computes a forward step with the given data
        Parameters
        ----------
        inputs : tf.Tensor
            the input for the model
        training : bool
            true if call has been made from train_step, which tells the dropout layer to be active 
        Returns
        ----------
        x : tf.Tensor
            the output of the model
        """
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x


if __name__ == '__main__':
    model = Sequential(
        [
        Conv2D(filters=64, kernel_size=(3,3)),
        LeakyReLU(),
        Dropout(0.3),
        Conv2D(filters=128, kernel_size=(3,3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPool2D(),
        Conv2D(filters=128, kernel_size=(3,3)),
        LeakyReLU(),
        Dropout(0.3),
        MaxPool2D(),
        Conv2D(filters=64, kernel_size=(3,3)),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.3),
        MaxPool2D(),
        Conv2D(filters=32, kernel_size=(3,3)),
        BatchNormalization(),
        LeakyReLU(),
        MaxPool2D(),
        Flatten(),
        Dense(4, activation='softmax')
        ], name='ConvNet2'  
    ) 
    x = tf.ones((1, 630, 50,25))
    y = model(x)
    model.summary()

