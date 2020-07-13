import tensorflow as tf
from tensorflow import keras
from os import listdir
import generator as gen
import discriminator as discrim


if __name__ == "__main__":
    INPUT_SIZE = 100

    generator = gen.model(INPUT_SIZE)
    discriminator = discrim.model(INPUT_SIZE)
