import tensorflow as tf
import generator as gen
import discriminator as dis
import matplotlib.pyplot as plot
import numpy as np
from tensorflow import keras
from os import listdir
from os.path import join
from VanImages import refresh_dir
from argparse import ArgumentParser
from PIL import Image

parser = ArgumentParser(
    description="Makes and trains GANs for painting generation")
parser.add_argument("--refresh", nargs=1)
args = parser.parse_args()

# Main Config
OUTDIR = "res"
EPOCHS = 25
IMAGE_URLS = listdir(OUTDIR)
TRAINING_SIZE = len(IMAGE_URLS)
BATCH = TRAINING_SIZE // 32

INPUT_SIZE = 100
# Width * Height * Channels
INPUT_SHAPE = (128, 128, 3)

if args.refresh != None:
    shape = (int(args.refresh[0]), int(args.refresh[0]), 3)
    if shape[0] / 32. % 1 != 0:
        raise Exception("Edges not multiples of 32")
    INPUT_SHAPE = shape
    refresh_dir(OUTDIR, (shape[0], shape[0]))

# Get Data
data = []
for url in IMAGE_URLS:
    image = Image.open(join(OUTDIR, url))
    data.append(np.asarray(image))
# Reshapes data to be (data_size, img, img, channel)
data = np.reshape(data, (-1, INPUT_SHAPE[0], INPUT_SHAPE[0], 3))
# Pixel values from -1 to 1
data = data.astype(np.float32)
data = data / 127.5 - 1

# Make models
generator = gen.model(INPUT_SHAPE, INPUT_SIZE)
discriminator = dis.model(INPUT_SHAPE)

# The optimizers that will adjust the models
gen_optimizer = keras.optimizers.Adam(1.3e-4, .48)
dis_optimizer = keras.optimizers.Adam(1.3e-4, .48)


# Compiled function
@tf.function
def step(img_list):
    pass


def train(img_list, epochs):
    pass


train(data, EPOCHS)
