import tensorflow as tf
import tensorflow.keras.layers as layers
# from utils import * 

# Generator - desired image size 28x28x1
class Generator:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.optimizer = self.optimizer()
        self.create_model()

    def create_model(self):
        self.model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Reshape((7,7,256)))
        assert self.model.output_shape == (None, 7, 7, 256)

        self.model.add(layers.Conv2DTranspose(128,(5,5), strides=(1,1), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 7, 7, 128)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(64,(5,5), strides=(2,2), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 14, 14, 64)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)

        return self.model

    def loss(self,fake):
        entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = entropy(tf.ones_like(fake), fake)
        return loss

    def optimizer(self,epsilon=1e-4):
        return tf.keras.optimizers.Adam(epsilon)