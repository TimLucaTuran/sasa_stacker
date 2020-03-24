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

class RunningAvg(tf.keras.layers.Layer):
    def __init__(self, N):
        self.N = N
        super(RunningAvg, self).__init__()

    def _running_avg_2d(self, x):
        #print("[INFO] numpy input:", x.shape)
        batch_dim = x.shape[0]
        z_dim = x.shape[2]
        for i in range(batch_dim):
            for j in range(z_dim):
                x[i,:,j] = np.convolve(x[i,:,j], np.ones((self.N,))/self.N, mode='same')
        return x

    def call(self, input):
        #print("[INFO] Avg Input:", input)
        input_shape = input.get_shape()

        out_tensor = tf.numpy_function(
            func = self._running_avg_2d,
            inp = [input],
            Tout = float,
        )
        out_tensor.set_shape(input_shape)
        return out_tensor
#%%
gen = batch_generator("data/bili_validation")
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
with CustomObjectScope({'loss': mse_with_changable_weight(continuous_out_loss)}):
            old_model = load_model("data/models/more_kernels__forward.h5")

w = np.asarray([
    [
    [
    [0,0],
    [0,0],
    [0,0]
    ],
    [
    [0,0],
    [0,0],
    [0,0]
    ]
    ]
    ])
old_model.layers
l = old_model.layers[9].get_weights()[0]
l.shape
RunningAvg = Conv1D(
    filters = 2,
    kernel_size = 3,
    padding='same',
    name="RunningAvg",
    )
RunningAvg.trainable = False
x = RunningAvg(old_model.output)
model = Model(inputs=old_model.input, outputs=x)

l = RunningAvg.get_weights()
l[0].shape

RunningAvg.set_weights(w)

opt = Adam()
model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])

spec, design = gen.__next__()
design[0] = design[0].astype(float)
spec_ = model(design)
wav = np.linspace(0.4, 1.2, 160)

for n in range(20, 30):
    plt.plot(wav, spec[n,:,1])
    plt.plot(wav, spec_[n,:,1])
    plt.show()

fig, ax = plt.subplots()
ax.plot(wav, x[22,:,1])
ax.plot(wav, x_[22,:,1])
fig.savefig("data/plots/combined2.pdf")

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
