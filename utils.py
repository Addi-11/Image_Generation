import tensorflow as tf
# import imageio
# import glob
import matplotlib.pyplot as plt
import os

def create_dataset(train_images, buffer_size, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

def normalise(train_images):
    # reshaping to the desired input shape
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    # normalizing
    train_images = (train_images - 127.5) / 127.5
    return train_images

def generate_and_save_images(model, epoch, test_input):
    # `training` is set to False so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def create_ckpt_dir(generator, discriminator):
    ckpt_dir = './training_checkpoints'
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer, discriminator_optimizer=discriminator.optimizer, 
                                 generator=generator.model, discriminator=discriminator.model)
    return checkpoint, ckpt_dir, ckpt_prefix