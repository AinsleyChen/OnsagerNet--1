import tensorflow as tf
import sys
import os
import inspect
import random
import numpy as np

# ------------------------------------------------------------------ #
#                         Serialization Utils                        #
# ------------------------------------------------------------------ #


def get_custom_objs(modules=[
    'onsager_rnn.onsagernet',
    'onsager_rnn.layers',
    'onsager_rnn.initializers',
]):
    """Get custom objects in modules

    HACK: This may not be platform agnostic?
    Consider also using ``tf.keras.utils.CustomObjectScope`` and
    associated functions
    """
    import onetlib.onsagernet
    import onetlib.layers  # noqa F401
    import onetlib.initializers  # noqa F401
    custom_objs_dict = {}
    for module in modules:
        module_info = inspect.getmembers(sys.modules[module], inspect.isclass)
        custom_objs_dict.update(dict(module_info))
    return custom_objs_dict


def set_gpu_growth():
    """Set GPU to allow growth

    Reference: https://www.tensorflow.org/guide/gpu
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def set_random_seed(seed, disable_gpu=True):
    """Set random seed for reproducibility

    For strict reproducibility, can set ``disable_gpu=True``
    to disable GPU usage.

    Reference: https://tinyurl.com/y5nttbqd
    """
    if disable_gpu:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
