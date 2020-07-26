import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import entropy

# Generator - desired image size 28x28x1
class Generator:
    def model(self, *args):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7,7,256)))
        assert model.output_shape == (None, 7, 7, 256)

        model.add(layers.Conv2DTranspose(128,(5,5), strides=(1,1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64,(5,5), strides=(2,2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def loss(self,fake):
        return entropy(tf.ones_like(fake), fake)

    def optimizer(self,epsilon=1e-4):
        return tf.keras.optimizers.Adam(epsilon)