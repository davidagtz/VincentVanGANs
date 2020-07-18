import tensorflow as tf
import generator as gen
import discriminator as dis
import matplotlib.pyplot as plot
import numpy as np
from tensorflow import keras, GradientTape
from os import listdir, mkdir
from os.path import join, exists, isfile, isdir
from VanImages import refresh_dir
from argparse import ArgumentParser
from PIL import Image
import time
from config import config_write, config_read

tf.get_logger().setLevel("ERROR")


def time_string(seconds):
    hours = seconds // 3600
    seconds = seconds % 3600

    minutes = seconds // 60
    seconds = seconds % 60

    return "{}:{:02}:{:>05.2f}".format(int(hours), int(minutes), seconds)


parser = ArgumentParser(
    description="Makes and trains GANs for painting generation")
parser.add_argument("--refresh", type=int,
                    help="Redownload images and resize them to parameter")
parser.add_argument("indirs", nargs="*", default=[])
parser.add_argument("--outdir", "-o", default="assets/output")
parser.add_argument("--epochs", "-e", default=50, type=int)
parser.add_argument("--load", action="store_true",
                    help="Load a model using the outdir")
parser.add_argument("--momentum", "-p", default=None, type=float,
                    help="BatchNormalization Momentum")
parser.add_argument("--optimizer", nargs=2, type=float, default=None,
                    help="alpha and beta for optimizer")
parser.add_argument("--separate", action="store_true")
parser.add_argument("--every", default=None, type=int,
                    help="when to save epoch progress")
parser.add_argument("--no-log", action="store_true")
parser.add_argument("--stop", action="store_true")
args = parser.parse_args()

# Main Config
OUTDIR = args.outdir
INDIRS = args.indirs
if not exists(OUTDIR):
    mkdir(OUTDIR)

cf = config_read(OUTDIR, momentum=.9, alpha=1e-6,
                 beta=5e-2, every=1, indirs=INDIRS)
opt = args.optimizer

INDIRS = cf["PATHS"]

if len(INDIRS) == 0 and not args.load:
    raise Exception("No file directories given")

EPOCHS = args.epochs
IMAGE_URLS = []
TRAINING_SIZE = len(IMAGE_URLS)
BATCH = max(1, TRAINING_SIZE // 32)
STARTSTEP = 0
ALPHA = opt[0] if opt is not None else float(cf.get("alpha"))
BETA = opt[1] if opt is not None else float(cf.get("beta"))
MOMENTUM = args.momentum if args.momentum is not None else float(
    cf.get("momentum"))
EVERY = args.every if args.every is not None else int(cf.get("every"))
STOP = args.stop
LOG = ""


def log(message, stdout=True):
    global LOG
    global args
    if not args.no_log:
        LOG += message + "\n"
    if stdout:
        print(message)


SEED = 100
# Width * Height * Channels
INPUT_SHAPE = (128, 128, 3)

# Print to config
config_write(OUTDIR, momentum=MOMENTUM, alpha=ALPHA,
             beta=BETA, every=EVERY, indirs=INDIRS)

if not STOP:
    log("Will train the discriminator...")
else:
    log("Will not train the discriminator...")

log(f"Alpha: {ALPHA}", stdout=False)
log(f"Beta: {BETA}", stdout=False)
log(f"Momentum: {MOMENTUM}", stdout=False)
log(f"Epochs: {EPOCHS}", stdout=False)
log(f"Every: {EVERY}", stdout=False)
log(f"Batch: {BATCH}", stdout=False)
log(f"Inputs:\n{"\n\t".join(INDIRS)}", stdout=False)


if args.refresh != None:
    shape = (args.refresh, args.refresh, 3)
    if shape[0] / 32. % 1 != 0:
        raise Exception("Edges not multiples of 32")
    INPUT_SHAPE = shape

    refresh_dir("assets/res", (shape[0], shape[1]), separate=args.separate)

for url in INDIRS:
    IMAGE_URLS += [join(url, file) for file in listdir(url)]

# Get Data
data = []
for url in IMAGE_URLS:
    image = Image.open(url)
    t = np.asarray(image)
    if t.shape != INPUT_SHAPE:
        print(f"Warning: { url } was skipped")
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

generator = None
discriminator = None
if args.load:
    models = [x for x in listdir(OUTDIR) if (
        isdir(join(OUTDIR, x)) and x.endswith(".model"))]
    models.sort()

    gen_name = [x for x in models if x.startswith("generator")]
    gen_name = f"generator-{len(gen_name) - 1}.model"

    dis_name = [x for x in models if x.startswith("discriminator")]
    dis_name = f"discriminator-{len(dis_name) - 1}.model"

    generator = tf.keras.models.load_model(join(OUTDIR, gen_name))
    discriminator = tf.keras.models.load_model(
        join(OUTDIR, dis_name))

    files = listdir(OUTDIR)
    steps = [x for x in files if isfile(
        join(OUTDIR, x)) and x.endswith(".png")]

    max_num = 0
    for img in steps:
        max_num = max(max_num, int(
            img[img.index("-") + 1: img.rindex(".")]))
    STARTSTEP = max_num + 1
else:
    # Make models
    generator = gen.model(INPUT_SHAPE, SEED, MOMENTUM=MOMENTUM)
    discriminator = dis.model(INPUT_SHAPE, MOMENTUM=MOMENTUM)

log(f"Starting Epoch: {STARTSTEP}", stdout=False)

# The optimizers that will adjust the models
gen_optimizer = keras.optimizers.Adam(ALPHA, BETA)
dis_optimizer = keras.optimizers.Adam(ALPHA, BETA)

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
        dis_gradients = 0
        if not STOP:
            dis_gradients = dis_tape.gradient(
                dis_loss, discriminator.trainable_variables)

        # Apply gradients
        gen_optimizer.apply_gradients(
            zip(gen_gradients, generator.trainable_variables))
        if not STOP:
            dis_optimizer.apply_gradients(
                zip(dis_gradients, discriminator.trainable_variables))

        return gen_loss, dis_loss


def train(img_list, epochs):
    TRAIN_START = time.time()
    EX_SEED = np.random.normal(0, 1, (IMAGE_COLS * IMAGE_ROWS, SEED))

    for epoch in range(STARTSTEP, STARTSTEP + epochs):
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

        message = f"Epoch { epoch }, gen loss = { gen_loss }, dis loss = { dis_loss }, { time_string(EPOCH_ELAPSED) }"
        log(message)

        if (epoch - STARTSTEP) % EVERY == 0 or epoch - STARTSTEP == epochs - 1:
            save_step(epoch, EX_SEED)

    ELAPSED = time.time() - TRAIN_START
    print(f"\nTraining time: {time_string(ELAPSED)}")


train(dataset, EPOCHS)

i = 0
while exists(join(OUTDIR, f"generator-{i}.model")):
    i += 1
generator.save(join(OUTDIR, f"generator-{i}.model"))
discriminator.save(join(OUTDIR, f"discriminator-{i}.model"))

if not args.no_log:
    with open(join(OUTDIR, f"session-{i}.log"), "w") as file:
        file.write(log)
