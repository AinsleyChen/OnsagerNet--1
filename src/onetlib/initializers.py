import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import Identity
from math import sqrt


class ScaledGlorotUniform(GlorotUniform):
    def __init__(self, scale=1.0e-3, **kwargs):
        super(ScaledGlorotUniform, self).__init__(**kwargs)
        self.scale = scale

    def __call__(self, shape, dtype=None):
        values = super(ScaledGlorotUniform, self).__call__(shape=shape,
                                                           dtype=dtype)
        return values * self.scale

    def get_config(self):
        config = super(ScaledGlorotUniform, self).get_config()
        config.update({'scale': self.scale})
        return config


class FlattenedIdentity(Identity):
    def __call__(self, shape, dtype=None):
        linear_size = int(sqrt(shape[0]))
        values = super(FlattenedIdentity,
                       self).__call__(shape=(linear_size, linear_size),
                                      dtype=dtype)
        return tf.reshape(values, shape=(-1, ))
