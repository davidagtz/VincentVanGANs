import tensorflow as tf
from tensorflow import keras
from os import listdir
import generator as gen
import discriminator as discrim
import matplotlib.pyplot as plot

# optimizer = keras.optimizers.Adam(1.5e-4, 0.5)


def genTest(generator):
    noise = tf.random.normal([1, INPUT_SIZE])

    gen_img = generator(noise, training=False)[0, :, :, 0]
    return plot.imshow(gen_img)


if __name__ == "__main__":
    INPUT_SIZE = 100
    # Width * Height * Channels
    INPUT_SHAPE = (150, 150, 3)

    generator = gen.model(INPUT_SIZE)
    discriminator = discrim.model(INPUT_SHAPE)

    image = genTest(generator)
    plot.show()
