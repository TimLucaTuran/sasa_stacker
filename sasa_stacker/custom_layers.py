import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Conv1D
from tensorflow.keras.utils import CustomObjectScope

#Running average layer
def avg_init(shape, dtype=tf.float32):
    w = np.zeros(shape)

    for i in range(shape[2]):
        w[:,i,i] = np.ones(shape[0])/shape[0]
    return w


def RunningAvg(*args, **kwargs):
    layer = Conv1D(
        *args,
        **kwargs,
        use_bias=False,
        kernel_initializer=avg_init,
    )
    layer.trainable = False
    return layer

#Padding layer for stride 2 Conv1DTranspose
def stride_two_pad_output_shape(x):
    xs = x.shape
    return tf.TensorShape([xs[0], 2*xs[1], xs[2]])

def stride_two_pad(x):
    shape = stride_two_pad_output_shape(x)

    pad = tf.zeros_like(x)
    res = tf.stack([x, pad], 2)
    return tf.reshape(res, [-1, shape[1], shape[2]])

def ZeroPadding1DStride2():
    return Lambda(
        stride_two_pad,
        output_shape=stride_two_pad_output_shape)

def load_inverse_from_combined(combined_path):
    with CustomObjectScope({'avg_init': avg_init}):
                combined_model = load_model("data/models/combined.h5")

    discrete_out = combined_model.get_layer('discrete_out').output
    continuous_out = combined_model.get_layer('continuous_out').output
    inverse_model = Model(inputs=combined_model.input, outputs=[discrete_out, continuous_out])
    return inverse_model
