import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.keras.layers import deserialize
from onetlib.utils import get_custom_objs

# ------------------------------------------------------------------ #
#                             Base Layer                             #
# ------------------------------------------------------------------ #


class BaseLayer(Layer):
    """The Base Layer for OnsagerNet implementations.

    Additional features on top of ``Layer`` class:
    * Automatic serialization of external objects passed to the constructor,
    which is stored as a list of names in class attribute
    ``BaseLayer.external_layers``.
    * For subclassing, if the constructor needs externally provided layers
    as arguments, their precise names should be stored in this attribute.
    * For subclassing, the ``get_config`` method only needs to serialze
    attributions required for initiliazation *other* than the
    ``external_layers``
    * See subclassing examples below
    """

    external_layers = []

    def get_config(self):
        config = super(BaseLayer, self).get_config()
        for name in self.external_layers:
            layer = getattr(self, name)
            config.update({
                name: {
                    'class_name': layer.__class__.__name__,
                    'config': layer.get_config()
                }
            })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        external_layer_config = {}
        custom_obj_dict = get_custom_objs()
        with custom_object_scope(custom_obj_dict):
            for name in cls.external_layers:
                external_layer = deserialize(config.pop(name),
                                             custom_objects=custom_objects)
                external_layer_config.update({name: external_layer})
        layer = cls(**external_layer_config, **config)
        return layer


# ------------------------------------------------------------------ #
#                          Activation layers                         #
# ------------------------------------------------------------------ #


class RePU(BaseLayer):
    """Rectified Power Unit Activation

    z->max(0,z)^p
    """
    def __init__(self, p=2, **kwargs):
        """Initializer

        :param p: power, defaults to 2
        :type p: int, optional
        """
        super(RePU, self).__init__(**kwargs)
        self.p = p

    def call(self, inputs):
        return tf.pow(tf.nn.relu(inputs), self.p)

    def get_config(self):
        config = super(RePU, self).get_config()
        config.update({'p': self.p})
        return config


class ShiftedRePU(RePU):
    """Shifted Rectified Power Unit Activation

    z->max(0,z)^p - max(0,z-0.5)^p
    """
    def call(self, inputs):
        outputs = super(ShiftedRePU, self).call(inputs)
        shifted_outputs = super(ShiftedRePU, self).call(inputs - 0.5)
        return outputs - shifted_outputs
    




# ------------------------------------------------------------------ #
#                        Miscellaneous layers                        #
# ------------------------------------------------------------------ #


class RegularizedSquare(BaseLayer):
    """Regularized Square Summation Layer

    Takes as input two tensors [x, z] and returns
    beta * sum_j x[:, j]**2 + sum_j z[:, j]**2
    """
    def __init__(self, beta=0.1, **kwargs):
        """Initializer

        :param beta: regularization parameter, defaults to 0.1
        :type beta: float, optional
        """
        super(RegularizedSquare, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        x, z = inputs
        x_sum = tf.reduce_sum(x**2, axis=1, keepdims=True)
        z_sum = tf.reduce_sum(z**2, axis=1, keepdims=True)
        return self.beta * x_sum + z_sum

    def get_config(self):
        config = super(RegularizedSquare, self).get_config()
        config.update({'beta': self.beta})
        return config


class SymmAntiDecomposition(BaseLayer):
    """Symmetric-Antisymmetric Decomposition Layer

    Takes as inputs a matrix A of size [Batch, n_dim, n_dim]
    Returns (M, W) where
    * M = LL^T
    * W = U - U^T
    where L and U are the lower and upper triangles of A
    """
    def __init__(self, n_dim, **kwargs):
        """Initializer

        :param n_dim: dimension of the matrix
        :type n_dim: int
        """
        super(SymmAntiDecomposition, self).__init__(**kwargs)
        self.n_dim = n_dim

    def call(self, inputs):
        A = tf.reshape(inputs, [-1, self.n_dim, self.n_dim])
        lower_triangle = tf.linalg.band_part(A, -1, 0)
        upper_triangle = tf.linalg.band_part(A, 0, -1)
        symmetric = lower_triangle @ tf.transpose(lower_triangle, [0, 2, 1])
        antisymmetric = upper_triangle - tf.transpose(upper_triangle,
                                                      [0, 2, 1])
        return symmetric, antisymmetric

    def get_config(self):
        config = super(SymmAntiDecomposition, self).get_config()
        config.update({'n_dim': self.n_dim})
        return config


class OnsagerCombination(BaseLayer):
    """Combination Layer for OnsagerNet

    Takes as input a tuple [M, W, g, f] and outputs
    - (M + W) g - alpha * g + f
    """
    def __init__(self, alpha, **kwargs):
        """Initializer

        :param alpha: regularizer
        :type alpha: float
        """
        super(OnsagerCombination, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        m, w, g, f = inputs
        return -tf.einsum('ijk,ik->ij', m + w, g) - self.alpha * g + f

    def get_config(self):
        config = super(OnsagerCombination, self).get_config()
        config.update({'alpha': self.alpha})
        return config


class GradientLayer(BaseLayer):
    """Gradient Layer

    Takes gradient of ``func`` with respect to the inputs.
    That is: GradientLayer(f) is a layer that maps x->df/dx
    """
    external_layers = ['func']

    def __init__(self, func, **kwargs):
        """Initializer

        :param func: layer to be differentiated
        :type func: BaseLayer
        """
        super(GradientLayer, self).__init__(**kwargs)
        self.func = func

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.func(inputs)
        grads = tape.gradient(outputs, inputs, unconnected_gradients='zero')
        return grads


# ------------------------------------------------------------------ #
#                         Integration layers                         #
# ------------------------------------------------------------------ #


class FwEulerLayer(BaseLayer):
    """Forward Euler Integration Layer

    Computes the forward Euler integration step.
    Take as inputs [x, fx] and outputs x + dt * fx
    """
    def __init__(self, dt, **kwargs):
        """Initializer

        :param dt: time step
        :type dt: float
        """
        super(FwEulerLayer, self).__init__(**kwargs)
        self.dt = dt

    def call(self, inputs):
        x, fx = inputs
        return x + self.dt * fx

    def get_config(self):
        config = super(FwEulerLayer, self).get_config()
        config.update({'dt': self.dt})
        return config
