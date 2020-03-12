import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MaxPooling1D, Dropout, Conv1D, GlobalMaxPooling1D, Reshape, BatchNormalization, Flatten
from tensorflow.keras.losses import mean_squared_error, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import numpy as np
from fit import SingleLayerInterpolator, calculate_spectrum, classify_output
from train import batch_generator
from data_gen import LabelBinarizer
from hyperparameters import MODEL_DISCRETE_OUTPUTS
#%%
lb = LabelBinarizer()
gen = batch_generator("data/bili_validation")

def my_func(x):
    sep = MODEL_DISCRETE_OUTPUTS
    p1, p2, p_stack = classify_output(x[:sep], x[sep:], lb)
    return np.zeros(x.shape)

class ForwardPass(tf.keras.layers.Layer):

  def __init__(self):
    super(ForwardPass, self).__init__()

  def call(self, inputs):
    return tf.py_function(func=my_func, inp=[inputs], Tout=tf.float32)


inp = Input(shape=(18))
x = Dense(10)(inp)
x = ForwardPass()(x)


a = gen.__next__()[1]
a[1].shape
model = Model(inputs=inp, outputs=x)

model(a)
np.zeros()
