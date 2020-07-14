from tensorflow import ones_like, zeros_like, keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def loss(fake):
    return cross_entropy(ones_like(fake), fake)


def model(shape, size):
    model = Sequential()

    MULTIPLE = shape[0] // 32

    model.add(Dense(4*4*256, activation="relu", input_dim=size))
    # Unflatten Layer
    model.add(Reshape((4, 4, 256)))

    # First convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Second convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Third convolutional layer
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Scaled CNN Layer
    if MULTIPLE > 1:
        model.add(UpSampling2D(size=(MULTIPLE, MULTIPLE)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

    # Final convolutional layer
    # 3 is for channels
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model
