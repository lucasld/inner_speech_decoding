import tensorflow as tf

class SimpleConv(tf.keras.models.Model):
    def __init__(self, conv_size=[64, 16], drop_out=0.6):
        """ A simple Convolutional Network made up of two Convolutional Blocks
        init: constructor of model
        :param conv_size: Array of filter size of the two convolutional layers
        :type conv_size: array of intergers
        :param drop_out: Dropout rate for the dropout layer
        :type drop_out: Float between 0 and 1
        call: performs forward pass of model
        """
        super(SimpleConv, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=conv_size[0], kernel_size=(3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(drop_out),
            tf.keras.layers.Conv2D(filters=int(conv_size[1]), kernel_size=(3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((1, 1)),
            tf.keras.layers.Dropout(drop_out / 2),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ]

    @tf.function
    def call(self, inputs) -> tf.Tensor:
        """ Computes a forward step with the given data
        Parameters
        ----------
        inputs : tf.Tensor
            the input for the model
        Returns
        ----------
        x : tf.Tensor
            the output of the model
        """
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x


class SimpleFF(tf.keras.models.Model):
    """
    Simple Feed Forward Network of 3 Dense Layers connected by Dropout layers Followed by an output layer
    """
    def __init__(self, input_dim=[128, 64, 16]):
        """
        init: constructor of model
        :param input_dim: Array of the units of the dense layers
        :type input_dim: Array of integers
        call: performs forward pass of model
        """
        super(SimpleFF, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(input_dim[0], activation='relu'),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(input_dim[1], activation='relu'),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(input_dim[2], activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ]

    @tf.function
    def call(self, inputs) -> tf.Tensor:
        """ Computes a forward step with the given data
        Parameters
        ----------
        inputs : tf.Tensor
            the input for the model
        Returns
        ----------
        x : tf.Tensor
            the output of the model
        """
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x
