#standard lib modules
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sqlite3
import argparse
import pickle
import cProfile
import matplotlib

#NN modules
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.utils import CustomObjectScope
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

#Self written modules
from sasa_db.crawler import Crawler
from sasa_phys.stack import *
from hyperparameters import *

INIT_LR = 1e-3
#%%
def moving_average(x):

def create_model():
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

    model = Model(inputs=inp, outputs=[discrete_out, continuous_out])
    return model

def create_forward_model():
    #merge the output of the inverse network
    dis_in = Input(shape=MODEL_DISCRETE_OUTPUTS)
    cont_in = Input(shape=MODEL_CONTINUOUS_OUTPUTS)
    x = Concatenate()([dis_in, cont_in])
    x = Dense(20*128)(x)
    x = Reshape((20,128))(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x) #40,64

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Lambda(lambda x: )()
    x = UpSampling1D()(x) #80,64

    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x) #160,128

    x = Conv1D(2, 3, activation='linear', padding='same')(x) #160,2
    x = BatchNormalization()(x)
    model = Model(inputs=[dis_in, cont_in], outputs=x)
    return model

class LossWeightsChanger(tf.keras.callbacks.Callback):
    def __init__(self, continuous_out_loss):
        self.continuous_out_loss = continuous_out_loss

    def on_epoch_end(self, epoch, logs={}):
        print("[INFO] current weight:", self.continuous_out_loss)
        print("[INFO] discrete_loss:", logs["discrete_out_loss"])
        print("[INFO] discrete_loss:", logs["continuous_out_loss"])
        self.continuous_out_loss = (self.continuous_out_loss *
            logs["discrete_out_loss"]/logs["continuous_out_loss"])

class BatchUpdater(tf.keras.callbacks.Callback):
    def __init__(self, batch_X):
        self.batch_X = batch_X

    def on_batch_begin(self, batch, logs={}):
        print("[INFO] batch[0].shape", batch[0].shape)
        self.batch_X = batch[0]

def mse_with_changable_weight(loss_weight):
    def loss(y_true, y_pred):
        loss_val = mean_squared_error(y_true, y_pred)
        return loss_weight*loss_val

    return loss

def batch_generator(batch_dir):
    """
    Just load the batches created by data_gen.py
    """

    inp_batches = []
    while True:
        #reset x_batches once are batches are used up
        if len(inp_batches) == 0:
            inp_batches = os.listdir(f"{batch_dir}/X")

        idx = random.randint(0, len(inp_batches)-1)
        batch = inp_batches[idx][:-4]#[:-4] to remove the .npy

        x = np.load(f"{batch_dir}/X/{batch}.npy")
        discrete_out = np.load(f"{batch_dir}/Y/{batch}.npy")

        with open(f"{batch_dir}/params/{batch}.pickle", "rb") as f:
            params = pickle.load(f)
        continuous_out = np.zeros((BATCH_SIZE, MODEL_CONTINUOUS_OUTPUTS))

        for i in range(BATCH_SIZE): #needs to be generalized
            layer1, layer2, stack = params[i]

            continuous_out[i,0] = layer1["width"]
            continuous_out[i,1] = layer1["length"]
            continuous_out[i,2] = layer1["thickness"]
            continuous_out[i,3] = layer1["periode"]

            continuous_out[i,4] = layer2["width"]
            continuous_out[i,5] = layer2["length"]
            continuous_out[i,6] = layer2["thickness"]
            continuous_out[i,7] = layer2["periode"]

            continuous_out[i,8] = stack["spacer_height"]
            continuous_out[i,9] = stack["angle"]

        del inp_batches[idx]


        yield (x, [discrete_out, continuous_out])

def forward_batch_generator(batch_dir):
    gen = batch_generator(batch_dir)
    while True:
        x, y = gen.__next__()
        yield y, x

#%%
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("model", metavar="m",
    	help="path to output model")
    ap.add_argument("batches", metavar="b",
    	help="path to directory containing the training batches")
    ap.add_argument("validation", metavar="v",
    	help="path to directory containing the validation batches")
    ap.add_argument("-p", "--params", default="data/smats/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    ap.add_argument("-log", "--log-dir", default="data/logs",
    	help="path to dir where the logs are saved")
    ap.add_argument("-n", "--new", action="store_true",
    	help="train a new model")
    ap.add_argument("-mt", "--model-type", default="inverse",
        help='["inverse", "forward", "both"] which kind of model to train')
    args = vars(ap.parse_args())
    print(args)



    if args["model_type"] == "inverse":

        print("[INFO] training inverse model...")
        continuous_out_loss = tf.Variable(1/40000)
        callbacks = [LossWeightsChanger(continuous_out_loss)]

        if args["new"]:
            model = create_model()
            opt = Adam() #decay=INIT_LR / EPOCHS lr=INIT_LR,
            losses = {
                'discrete_out' : 'binary_crossentropy',
                'continuous_out' : mse_with_changable_weight(continuous_out_loss),
                }
            loss_weights = {
                'discrete_out' : 1,
                'continuous_out' : 1,
                }
            model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
        else:
            #the scope is nessecary beacuse I used a custom loss for training
            with CustomObjectScope({'loss': mse_with_changable_weight(continuous_out_loss)}):
                model = load_model(args["model"])
        #Set the training generator
        generator = batch_generator
    elif args["model_type"] == "forward":

        print("[INFO] training forward model...")

        if args["new"]:
            model = create_forward_model()
            opt = Adam()
            model.compile(optimizer=opt, loss="mse", metrics=['mae'])
        else:
            model = load_model(args["model"])
        #Set the training generator
        generator = forward_batch_generator
        callbacks = []

    trainGen = generator(args["batches"])
    validationGen = generator(args["validation"])
    batch_count = len(os.listdir(f"{args['batches']}/X"))
    validation_count = len(os.listdir(f"{args['validation']}/X"))

    H = model.fit(
        trainGen,
	    steps_per_epoch=batch_count,
        validation_data=validationGen,
        validation_steps=validation_count,
        callbacks=callbacks,
        epochs=EPOCHS,
        )

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    print("[INFO] saving logs")
    model_name = args["model"].split("/")[-1][:-3]
    with open(f"{args['log_dir']}/{model_name}.pickle", "wb") as f:
        pickle.dump(H.history, f)

    print("[DONE]")
