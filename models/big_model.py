from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Flatten, Reshape, Dropout
from tensorflow.keras.layers import Dense, Activation
from math import log


def discriminator(shape, MOMENTUM=.9):
    model = Sequential()

    # First CNN Layer
    model.add(Conv2D(512, kernel_size=3, strides=2,
                     input_shape=shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    # Second CNN Layer
    model.add(Dropout(0.05))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(LeakyReLU(alpha=.2))

    # Third CNN Layer
    model.add(Dropout(0.05))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(LeakyReLU(alpha=.2))

    # Fourth CNN Layer
    model.add(Dropout(0.05))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(LeakyReLU(alpha=.2))

    # Third CNN Layer
    model.add(Dropout(0.05))
    model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(LeakyReLU(alpha=.2))

    # model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model


def generator(shape, size, MOMENTUM=.9):
    model = Sequential()

    MULTIPLE = shape[0] // 32

    model.add(Dense(4*4*256, activation="relu", input_dim=size))
    # Unflatten Layer
    model.add(Reshape((4, 4, 256)))

    # First convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(LeakyReLU(alpha=.2))

    # Second convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(LeakyReLU(alpha=.2))

    # Third convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(LeakyReLU(alpha=.2))

    # Scaled CNN Layer
    if MULTIPLE > 1:
        quotient = MULTIPLE
        while log(quotient, 2) % 1 != 0:
            model.add(UpSampling2D())
            model.add(Conv2D(256, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=MOMENTUM))
            model.add(LeakyReLU(alpha=.2))
            quotient /= 2

        model.add(UpSampling2D(size=(quotient, quotient)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=MOMENTUM))
        model.add(LeakyReLU(alpha=.2))

    # Final convolutional layer
    # 3 is for channels
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model
