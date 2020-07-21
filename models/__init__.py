from tensorflow import ones_like, zeros_like
from tensorflow.keras.losses import BinaryCrossentropy

__all__ = ["default", "new_model"]

cross_entropy = BinaryCrossentropy(from_logits=True)


def generator_loss(fake):
    return cross_entropy(ones_like(fake), fake)


def discriminator_loss(real, fake):
    rloss = cross_entropy(ones_like(real), real)
    floss = cross_entropy(zeros_like(fake), fake)
    return rloss + floss
