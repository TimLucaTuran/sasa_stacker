import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, MaxPooling1D, Concatenate, Dropout, Conv1D, GlobalMaxPooling1D, Reshape, BatchNormalization, Flatten
from tensorflow.keras.losses import mean_squared_error, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
import matplotlib.pyplot as plt

from fit import SingleLayerInterpolator, calculate_spectrum, classify_output
from train import batch_generator, mse_with_changable_weight
from data_gen import LabelBinarizer
from hyperparameters import *
from sasa_db.crawler import Crawler
import sqlite3
#%%
lb = LabelBinarizer()
gen = batch_generator("data/bili_validation")
continuous_out_loss = tf.Variable(1/40000)
#conn = sqlite3.connect("data/NN_smats.db")
#cursor = conn.cursor()
#c = Crawler(directory="data/smats", cursor=cursor)
#sli = SingleLayerInterpolator(c)

def my_func(*inputs):
    print(inputs[0].shape)
    conn = sqlite3.connect("data/NN_smats.db")
    cursor = conn.cursor()
    c = Crawler(directory="data/smats", cursor=cursor)
    sli = SingleLayerInterpolator(c)

    discrete_in = inputs[0]
    continuous_in = inputs[1]
    assert discrete_in.shape[0] == continuous_in.shape[0]
    batch_part_size = discrete_in.shape[0]

    output = np.zeros((batch_part_size,160,2))
    for i in range(batch_part_size):
        discrete_in = inputs[0][i]
        continuous_in = inputs[1][i]
        p1, p2, p_stack = classify_output(discrete_in, continuous_in, lb)
        spec = calculate_spectrum(p1, p2, p_stack, c, sli)
        output[i] = spec
    return output

class SasaLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SasaLayer, self).__init__()

    def build(self, input_shape):
        super(SasaLayer, self).build(input_shape)

    def call(self, inputs):
        print("[INFO] SasaLayer called with", inputs[0])
        out_tensor = tf.py_function(func=my_func, inp=inputs, Tout=tf.float32)
        if out_tensor.get_shape().rank is None:
            print("[INFO] corrected shape")
            out_tensor.set_shape(tf.TensorShape([None,160,2]))
        #print("[INFO] out_tensor dims:", out_tensor.get_shape().dims)
        return out_tensor
#%%

with CustomObjectScope({'loss': mse_with_changable_weight(continuous_out_loss)}):
            old_model = load_model("data/bili.h5")

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


opt = Adam()
model.summary()
model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])
a = gen.__next__()[0]
a.shape
b = model(a)
b.shape

n = 40
plt.plot(a[n,:,1])
plt.plot(b[n,:,1])


model.fit(a, a)

#%%

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)


inp = Input(shape=10)
x = MyLayer(15)(inp)
model = Model(inputs=inp, outputs=x)
model.summary()
#%%
