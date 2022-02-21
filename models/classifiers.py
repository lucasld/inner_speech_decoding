import tensorflow as tf

class ConvNet1(tf.keras.Model):
    def __init__(self, layer_list=[
        tf.keras.layers.Conv2D(filters=64, kernel_size=3),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=6),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(50, 'sigmoid'),
        tf.keras.layers.Dense(10, 'sigmoid'),
        tf.keras.layers.Dense(4, 'softmax'),
    ]) -> None:
        super(ConvNet1, self).__init__()
        self.layer_list = layer_list
    
    @tf.function
    def call(self, inputs) -> tf.Tensor:
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x