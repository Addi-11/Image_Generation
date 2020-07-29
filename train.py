import tensorflow as tf
from  utils import *
import time
from IPython import display

# @tf.function
def train_step(images, generator, discriminator, BATCH_SIZE=256, noise_dim=100):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator.model(noise, training=True)

      real_output = discriminator.model(images, training=True)
      fake_output = discriminator.model(generated_images, training=True)

      gen_loss = generator.loss(fake_output)
      disc_loss = discriminator.loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.model.trainable_variables)

    generator.optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
    discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))


def train(train_dataset, generator, discriminator,ckpt, ckpt_pre, EPOCHS=30):
  seed = tf.random.normal([16, 100])

  for epoch in range(EPOCHS):
    start = time.time()

    for image_batch in train_dataset:
        train_step(image_batch, generator, discriminator)

    # Producing images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator.model, epoch + 1, seed)

    # Saving checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        ckpt.save(file_prefix = ckpt_pre)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
