import tensorflow as tf
from tensorflow import keras
from os import listdir
import generator as gen
import discriminator as discrim
import matplotlib.pyplot as plot
from VanImages import refresh_dir
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(
        description="Makes and trains GANs for Van Gogh Painting generation")

    parser.add_argument("--refresh", nargs=2, default=None, required=False)

    return parser.parse_args()

# optimizer = keras.optimizers.Adam(1.5e-4, 0.5)


def genTest(generator):
    noise = tf.random.normal([1, INPUT_SIZE])

    gen_img = generator(noise, training=False)[0, :, :, 0]
    return plot.imshow(gen_img)


if __name__ == "__main__":
    INPUT_SIZE = 100
    # Width * Height * Channels
    INPUT_SHAPE = (128, 128, 3)

    args = get_args()
    if args.refresh != None:
        shape = (int(args.refresh[0]), int(args.refresh[1]), 3)
        if shape[0] / 32. % 1 != 0 or shape[0] != shape[1]:
            raise Exception("Either bounds are not square or a multiple of 32")
        INPUT_SHAPE = shape
        refresh_dir("res", (shape[0], shape[1]))

    generator = gen.model(INPUT_SHAPE, INPUT_SIZE)
    discriminator = discrim.model(INPUT_SHAPE)

    image = genTest(generator)
    plot.show()
