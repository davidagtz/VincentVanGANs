import tensorflow as tf
from tensorflow import keras
from os import listdir
import generator as gen
import discriminator as discrim

# optimizer = keras.optimizers.Adam(1.5e-4, 0.5)


if __name__ == "__main__":
    INPUT_SIZE = 100
    INPUT_SHAPE = (150, 150)

    generator = gen.model(INPUT_SIZE)
    discriminator = discrim.model(INPUT_SHAPE)
