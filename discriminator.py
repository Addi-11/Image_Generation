import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import entropy

# Discriminator
class Discriminator:
    def model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def loss(self, real, fake):
        real_loss = entropy(tf.ones_like(real), real)
        fake_loss = entropy(tf.zeros_like(fake), fake)
        loss = real_loss + fake_loss
        return loss
    
    def optimizer(self,epsilon=1e-4):
        return tf.keras.optimizers.Adam(epsilon)