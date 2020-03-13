import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, MaxPooling1D, Concatenate, Dropout, Conv1D, GlobalMaxPooling1D, Reshape, BatchNormalization, Flatten
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
import sqlite3
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
            old_model = load_model("data/bili.h5")



x = SasaLayer()(old_model.output)
x = Lambda(lambda x: K.stop_gradient(x))(x)
x = Activation('linear')(x)
model = Model(inputs=old_model.input, outputs=x)


opt = Adam()
model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])
x, y = gen.__next__()
x.shape
x_ = model(x)
x_.shape

n = 55
plt.plot(x[n,:,1])
plt.plot(x_[n,:,1])

model.fit(x, x)

model.summary()
for layer in model.layers:
    print(layer.name)
layer = model.get_layer(name="conv1d")
layer.kernel.name
#%%

inp = Input(shape=(MODEL_INPUTS, 2))
x = Conv1D(64, 5, activation='relu')(inp)
x = MaxPooling1D()(x)
x = Conv1D(64, 5, activation='relu')(x)
x = MaxPooling1D()(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D()(x)
x = Conv1D(256, 5, activation='relu')(x)
conv_out = GlobalMaxPooling1D()(x)

#discrete branch
x = Dense(256, activation='relu')(conv_out)
x = Dropout(0.5)(x)
discrete_out = Dense(MODEL_DISCRETE_OUTPUTS, activation='sigmoid', name='discrete_out')(x)

#continuous branch
x = Dense(256, activation='relu')(conv_out)
x = BatchNormalization()(x)
continuous_out = Dense(MODEL_CONTINUOUS_OUTPUTS, activation='linear', name='continuous_out')(x)

x = SasaLayer()([discrete_out, continuous_out])
model = Model(inputs=inp, outputs=x)

x = Lambda(call)(old_model.output)

def call(inputs):
    batch_size = inputs[0].get_shape().dims[0].value
    print("[INFO] SasaLayer called batch size", batch_size)
    out_tensor = tf.numpy_function(func=my_func, inp=inputs, Tout=tf.float32)
    out_tensor.set_shape(tf.TensorShape([batch_size,160,2]))

    return out_tensor
