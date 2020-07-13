from tensorflow import ones_like, zeros_like, keras
from tensorflow.keras import Sequential

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def loss(real, fake):
    rloss = cross_entropy(ones_like(real), real)
    floss = cross_entropy(zeros_like(fake), fake)
    return rloss + floss


def model(size):
    model = Sequential()

    optimizer = keras.optimizers.Adam(1.5e-4, 0.5)

    return model
