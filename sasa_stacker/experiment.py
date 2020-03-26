import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import CustomObjectScope
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import sys

from fit import SingleLayerInterpolator, calculate_spectrum, classify_output
from train import batch_generator, mse_with_changable_weight
from data_gen import LabelBinarizer
from hyperparameters import *
from sasa_db.crawler import Crawler
from testing import show_stack_info
import sqlite3
from utils import load_inverse_from_combined, stride_two_pad
#%%
a = tf.constant(np.arange(100).reshape(10,5,2), tf.float32)
res = stride_two_pad(a)
res[0,:,1]

b = tf.zeros_like(a)
a.shape[1]

ab = tf.stack([a,b], 2)
ab.shape
c = tf.reshape(ab,
               [10, 10, 2])
c.shape
c[0,:,1]
c[:,:-1]
def stride_two_pad(x):
    xs = x.shape
    shape = tf.TensorShape([xs[0], 2*xs[1], xs[2]])

    pad = tf.zeros_like(x)
    res = tf.stack([x, pad], 2)
    return tf.reshape(res, shape)
Lambda
#%%
gen = batch_generator("data/corrected_validation")
continuous_out_loss = tf.Variable(1/40000)
#conn = sqlite3.connect("data/NN_smats.db")
#cursor = conn.cursor()
#c = Crawler(directory="data/smats", cursor=cursor)
#sli = SingleLayerInterpolator(c)

def my_func(*inputs):
#   inputs = y
    print(inputs[0].shape, inputs[1].shape)
    lb = LabelBinarizer()
    conn = sqlite3.connect("data/NN_smats.db")
    cursor = conn.cursor()
    c = Crawler(directory="data/smats", cursor=cursor)
    sli = SingleLayerInterpolator(c)

    discrete_in, continuous_in = inputs
    assert discrete_in.shape[0] == continuous_in.shape[0]
    batch_part_size = discrete_in.shape[0]

    output = np.zeros((batch_part_size,160,2))
    for i in range(batch_part_size):
        p1, p2, p_stack = classify_output(discrete_in[i], continuous_in[i], lb)
        spec = calculate_spectrum(p1, p2, p_stack, c, sli)
        output[i] = spec

    return output

@tf.custom_gradient
def custom_op(inputs):
    batch_size = inputs[0].get_shape().dims[0].value
    print("[INFO] SasaLayer called batch size", batch_size)
    out_tensor = tf.numpy_function(func=my_func, inp=inputs, Tout=tf.float32)
    out_tensor.set_shape(tf.TensorShape([batch_size,160,2]))
    def custom_grad(dy):
        grad = dy
        return grad
    return out_tensor, custom_grad

def call_op(inputs):
    batch_size = inputs[0].get_shape().dims[0].value
    print("[INFO] SasaLayer called batch size", batch_size)
    out_tensor = tf.numpy_function(func=my_func, inp=inputs, Tout=tf.float32)
    out_tensor.set_shape(tf.TensorShape([batch_size,160,2]))
    return out_tensor

class SasaLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SasaLayer, self).__init__()

    def build(self, input_shape):
        super(SasaLayer, self).build(input_shape)

    def call(self, inputs):
        return call_op(inputs)


#%%
def avg_init(shape, dtype=tf.float32):
    w = np.zeros(shape)

    for i in range(shape[2]):
        w[:,i,i] = np.ones(shape[0])/shape[0]
    return w

with CustomObjectScope({'avg_init': avg_init}):
            old_model = load_model("data/models/ravg_ks8_forward.h5")
discrete_out = old_model.get_layer('discrete_out').output
continuous_out = old_model.get_layer('continuous_out').output
model = Model(inputs=old_model.input, outputs=[discrete_out, continuous_out])
model.summary()
def RunningAvg(filters, kernel_size, name):
    layer = Conv1D(
        filters = filters,
        kernel_size = kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer=avg_init,
        name=name,
    )
    layer.trainable = False
    return layer

x = RunningAvg(2, 8, "avg1")(old_model.output)
x = RunningAvg(2, 3, "avg2")(x)
x = RunningAvg(2, 3, "avg3")(x)
x = RunningAvg(2, 3, "avg4")(x)
model = Model(inputs=old_model.input, outputs=x)


model = old_model
opt = Adam()
model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])


spec, design = gen.__next__()
design[0] = design[0].astype(float)
spec_ = model(design)
wav = np.linspace(0.4, 1.2, 160)

s=1
for n in range(60, 90):
    plt.plot(wav, spec[n,:,s])
    plt.plot(wav, spec_[n,:,s])
    plt.show()


model.fit(x, x)
model.summary()
for layer in model.layers:
    print(layer.name)
layer = model.get_layer(name="conv1d")
layer.kernel.name

#%%
with CustomObjectScope({'loss': mean_squared_error}):
    model = load_model("data/models/corrected3_forward.h5")
model.layers[1].dtype
spec, design = gen.__next__()
design[0] = design[0].astype(float)
spec_ = model(design)


for i in range(40, 50):
    plt.plot(spec_[i,:,1])
    plt.plot(spec[i,:,1])
    plt.show()

Spec.shape
lb = LabelBinarizer()
stack = "data/square_validation/X/2020-03-10_11:52:38.181391.npy"
index = 0
batch_dir = "data/square_validation"
show_stack_info(model, stack, index, batch_dir)
