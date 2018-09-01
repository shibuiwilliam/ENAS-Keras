import keras
from keras import backend as K
from keras import initializers, regularizers


def get_weight_initializer(initializer=None, seed=None):
    if initializer is None:
        return initializers.he_normal()
    elif initializer == "lstm":
        return initializers.random_uniform(minval=-0.1, maxval=0.1)
    else:
        return initializer()


def get_weight_regularizer(regularizer=None, rate=1e-4):
    if regularizer is None:
        return regularizers.l2(rate)
    else:
        return regularizer(rate)
