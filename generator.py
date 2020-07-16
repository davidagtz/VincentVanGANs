from tensorflow import ones_like, zeros_like, keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation
from math import log

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def loss(fake):
    return cross_entropy(ones_like(fake), fake)


def model(shape, size, MOMENTUM=.9):
    model = Sequential()

    MULTIPLE = shape[0] // 32

    model.add(Dense(4*4*256, activation="relu", input_dim=size))
    # Unflatten Layer
    model.add(Reshape((4, 4, 256)))

    # First convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(Activation("relu"))

    # Second convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(Activation("relu"))

    # Third convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=MOMENTUM))
    model.add(Activation("relu"))

    # Scaled CNN Layer
    if MULTIPLE > 1:
        quotient = MULTIPLE
        while log(quotient, 2) % 1 != 0:
            model.add(UpSampling2D())
            model.add(Conv2D(128, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=MOMENTUM))
            model.add(Activation("relu"))
            quotient /= 2

        model.add(UpSampling2D(size=(quotient, quotient)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=MOMENTUM))
        model.add(Activation("relu"))

    # Final convolutional layer
    # 3 is for channels
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model
