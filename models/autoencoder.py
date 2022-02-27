import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, output_units) -> None:
        super(Encoder, self).__init__()
        self.layer_list = [
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2,
                                   padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2,
                                   padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2,
                                   padding='same', activation='relu')
        ]
    
    def call(self, input_):
        x = input_
        for layer in self.layer_list:
            x = layer(x)
        return x

class Decoder(tf.keras.Model):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.layer_list = [
            tf.keras.layers.Conv2DTranspose(8,3,2,padding='same'),
            tf.keras.layers.Conv2DTranspose(3,3,2,padding='same'),
            tf.keras.layers.Conv2DTranspose(1,3,2,padding='same')
        ]
    
    def call(self, input_):
        x = input_
        for layer in self.layer_list:
            x = layer(x)
        return x



class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self) -> None:
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = Encoder(7)
        self.decoder = Decoder()
    
    @tf.function()
    def call(self, input_):
        latent_space = self.encoder(input_)
        output = self.decoder(latent_space)
        return output