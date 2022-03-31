import tensorflow as tf
class DenseConv(tf.keras.models.Model):
    def __init__(self, conv_size=32, drop_out=0.6):
        """
        init: constructor of model
        call: performs forward pass of model
        """
        super(DenseConv, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=conv_size, kernel_size=(3,3), padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Dropout(drop_out),
            tf.keras.layers.Conv2D(filters=int(conv_size/2), kernel_size=(3,3), padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((1,1)),
            tf.keras.layers.Dropout(drop_out/2),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
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