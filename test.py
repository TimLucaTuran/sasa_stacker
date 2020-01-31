import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MaxPooling1D, Dropout, Conv1D, GlobalMaxPooling1D, Reshape, BatchNormalization, Flatten
from tensorflow.keras.losses import mean_squared_error, Huber
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import CustomObjectScope
from sklearn.preprocessing import MultiLabelBinarizer
#%%
"""MODEL_INPUTS = 160
mat = np.load('/home/tim/Uni/BA/stacker/data/pol_batches/X/2020-01-27_14:29:39.001003.npy')

mat.shape
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
discrete_out = Dense(8, activation='sigmoid', name='discrete_out')(x)

#continuous branch
x = Dense(256, activation='relu')(conv_out)
x = BatchNormalization()(x)
continuous_out = Dense(10, activation='linear', name='continuous_out')(x)

model = Model(inputs=inp, outputs=[discrete_out, continuous_out])
opt = Adam()#decay=INIT_LR / EPOCHS lr=INIT_LR,
losses = {
    'discrete_out' : 'binary_crossentropy',
    'continuous_out' : 'mse'
    }
loss_weights = {
    'discrete_out' : 1,
    'continuous_out' : 1,
    }
model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
model.summary()
y = model(mat)"""
#y[0,0,:]
x = np.linspace(-5, 5, 100)

matplotlib.use("Agg")
fig, ax = plt.subplots()
ax.grid(linestyle="--", linewidth=0.5)
ax.set_xlabel(r"Input to Node")
ax.set_ylabel(r"Activation")
ax.plot(x, relu(x), color="k", linewidth=0.7)
fig.savefig("bg_relu.pdf")
#plt.plot(x, sigmoid(x))
