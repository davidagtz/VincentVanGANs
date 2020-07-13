from tensorflow import ones_like, zeros_like, keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def loss(real, fake):
    rloss = cross_entropy(ones_like(real), real)
    floss = cross_entropy(zeros_like(fake), fake)
    return rloss + floss


def model(shape):
    model = Sequential()

    # First CNN Layer
    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape=shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    # Second CNN Layer
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=.2))

    # Third CNN Layer
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=.2))

    # Fourth CNN Layer
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=.2))

    # Third CNN Layer
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model
