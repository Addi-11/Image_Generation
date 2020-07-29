import tensorflow as tf
import tensorflow.keras.layers as layers
# from utils import entropy

# Discriminator
class Discriminator:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.optimizer = self.optimizer()
        # self.loss = None
        self.create_model()

    def create_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28,28,1]))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))

        return self.model

    def loss(self, real, fake):
        entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = entropy(tf.ones_like(real), real)
        fake_loss = entropy(tf.zeros_like(fake), fake)
        loss = real_loss + fake_loss
        return loss
    
    def optimizer(self,epsilon=1e-4):
        return tf.keras.optimizers.Adam(epsilon)