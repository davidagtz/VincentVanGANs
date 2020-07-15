import tensorflow as tf
import generator as gen
import discriminator as dis
import matplotlib.pyplot as plot
import numpy as np
from tensorflow import keras, GradientTape
from os import listdir
from os.path import join
from VanImages import refresh_dir
from argparse import ArgumentParser
from PIL import Image
import time


def time_string(seconds):
    hours = seconds // 3600
    seconds = seconds % 3600

    minutes = seconds // 60
    seconds = seconds % 60

    return "{}:{:02}:{:>05.2f}".format(hours, minutes, seconds)


parser = ArgumentParser(
    description="Makes and trains GANs for painting generation")
parser.add_argument("--refresh", nargs=1, type=int)
parser.add_argument("indirs", nargs="+")
parser.add_argument("--outdir", nargs=1, default="assets/output")
args = parser.parse_args()

# Main Config
INDIRS = args.indirs
OUTDIR = args.outdir
EPOCHS = 50
IMAGE_URLS = []
TRAINING_SIZE = len(IMAGE_URLS)
BATCH = max(1, TRAINING_SIZE // 32)

SEED = 100
# Width * Height * Channels
INPUT_SHAPE = (128, 128, 3)

for url in INDIRS:
    IMAGE_URLS += [join(url, file) for file in listdir(url)]


if args.refresh != None:
    shape = (args.refresh[0], args.refresh[0], 3)
    if shape[0] / 32. % 1 != 0:
        raise Exception("Edges not multiples of 32")
    INPUT_SHAPE = shape
    for url in INDIRS:
        refresh_dir(url, (shape[0], shape[1]))

# Get Data
data = []
for url in IMAGE_URLS:
    image = Image.open(url)
    t = np.asarray(image)
    if t.shape != INPUT_SHAPE:
        print(t.shape)
        print(url)
        continue
    data.append(np.asarray(image))

# Reshapes data to be (data_size, img, img, channel)
# -1 is adapative
data = np.reshape(data, (-1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
# Pixel values from -1 to 1
data = data.astype(np.float32)
data = data / 127.5 - 1
dataset = tf.data.Dataset.from_tensor_slices(
    data).shuffle(len(data)).batch(BATCH)

# Make models
generator = gen.model(INPUT_SHAPE, SEED)
discriminator = dis.model(INPUT_SHAPE)

# The optimizers that will adjust the models
alpha = 1e-6
beta = 5e-2
gen_optimizer = keras.optimizers.Adam(alpha, beta)
dis_optimizer = keras.optimizers.Adam(alpha, beta)

IMAGE_COLS = 4
IMAGE_ROWS = 2
MARGIN = INPUT_SHAPE[0] // 10


def save_step(name, noise):
    width = MARGIN + IMAGE_COLS * (MARGIN + INPUT_SHAPE[0])
    height = MARGIN + IMAGE_ROWS * (MARGIN + INPUT_SHAPE[1])
    images = np.full((height, width, INPUT_SHAPE[2]), 255, dtype=np.uint8)

    imgs = generator.predict(noise)
    imgs = 0.5 * imgs + 0.5

    i = 0
    for row in range(IMAGE_ROWS):
        for col in range(IMAGE_COLS):
            r_px = row * (INPUT_SHAPE[1] + MARGIN) + MARGIN
            c_px = col * (INPUT_SHAPE[0] + MARGIN) + MARGIN

            images[r_px:r_px + INPUT_SHAPE[1],
                   c_px:c_px + INPUT_SHAPE[0]] = imgs[i] * 255
            i += 1

    image = Image.fromarray(images)
    image.save(join(OUTDIR, f"step-{ name }.png"))


# Compiled function
@tf.function
def step(img_list):
    GEN_SEED = tf.random.normal([BATCH, SEED])

    # Having training=True makes dropout possible
    with GradientTape() as gen_tape, GradientTape() as dis_tape:
        # Generate Images
        fake_imgs = generator(GEN_SEED, training=True)

        # See discriminator performance
        real_dis = discriminator(img_list, training=True)
        fake_dis = discriminator(fake_imgs, training=True)

        # Measure losses
        gen_loss = gen.loss(fake_dis)
        dis_loss = dis.loss(real_dis, fake_dis)

        # Calculate gradients based off of loss and CNN
        gen_gradients = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        dis_gradients = dis_tape.gradient(
            dis_loss, discriminator.trainable_variables)

        # Apply gradients
        gen_optimizer.apply_gradients(
            zip(gen_gradients, generator.trainable_variables))
        dis_optimizer.apply_gradients(
            zip(dis_gradients, discriminator.trainable_variables))

        return gen_loss, dis_loss


def train(img_list, epochs):
    TRAIN_START = time.time()
    EX_SEED = np.random.normal(0, 1, (IMAGE_COLS * IMAGE_ROWS, SEED))

    for epoch in range(epochs):
        EPOCH_START = time.time()

        gen_loss_list = []
        dis_loss_list = []

        for batch in img_list:
            res = step(batch)
            gen_loss_list.append(res[0])
            dis_loss_list.append(res[1])

        gen_loss = sum(gen_loss_list) / len(gen_loss_list)
        dis_loss = sum(dis_loss_list) / len(dis_loss_list)

        EPOCH_ELAPSED = time.time() - EPOCH_START
        print(f"Epoch { epoch }, gen loss = { gen_loss }, \
            dis loss = { dis_loss }, { time_string(EPOCH_ELAPSED) }")

        save_step(epoch, EX_SEED)

    ELAPSED = time.time() - TRAIN_START
    print(f"\nTraining time: {time_string(ELAPSED)}")


train(dataset, EPOCHS)
generator.save(join(OUTDIR, "generator.model"))
discriminator.save(join(OUTDIR, "discriminator.model"))
