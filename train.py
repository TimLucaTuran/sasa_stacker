import sys
sys.path.insert(0, "../meta_material_databank")
sys.path.insert(0, "../SASA")


#standard modules
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import argparse
import pickle
import cProfile
import matplotlib
#NN modules
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MaxPooling1D, Dropout, Conv1D, GlobalMaxPooling1D, Reshape, BatchNormalization, Flatten
from tensorflow.keras.losses import mean_squared_error, Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import CustomObjectScope
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

#Self written modules
from crawler import Crawler
from stack import *

MODEL_INPUTS = 160
NUMBER_OF_WAVLENGTHS = MODEL_INPUTS
WAVLENGTH_START = 0.4
WAVLENGTH_STOP = 1.2
MODEL_DISCRETE_OUTPUTS = 8
MODEL_CONTINUOUS_OUTPUTS = 10
MODEL_DISCRETE_PREDICTIONS = {
    "particle_material" : ["Au", "Al"],
    "hole" : ["holes", "no holes"]
    }

BATCH_SIZE = 128
EPOCHS = 20
INIT_LR = 1e-3

#%%

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

class LossWeightsChanger(tf.keras.callbacks.Callback):
    def __init__(self, continuous_out_loss):
        self.continuous_out_loss = continuous_out_loss

    def on_epoch_end(self, epoch, logs={}):
        print("[INFO] current weight:", self.continuous_out_loss)
        self.continuous_out_loss = 2/logs["continuous_out_loss"]

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

def SASA_loss(lb, c, sli, batch_X=None):
    def loss(y_true, y_pred):
        loss_val = np.zeros(BATCH_SIZE)

        for i in range(BATCH_SIZE):
            discrete_out = y_pred[i,:8]
            continuous_out = y_pred[i,8:]

            p1, p2, p_stack = fit.classify_output(discrete_out, continuous_out, lb)
            spec = calculate_spectrum(p1, p2, p_stack, c, sli)

            loss_val[i] = fit.mean_squared_diff(spec, batch_X[i])

        return loss_val

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




#%%
if __name__ == '__main__':

    #%% construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batches", default="data/wire_batches",
    	help="path to directory containing the training batches")
    ap.add_argument("-v", "--validation", default="data/wire_validation",
    	help="path to directory containing the validation batches")
    ap.add_argument("-p", "--params", default="data/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    ap.add_argument("-m", "--model", default="data/stacker.h5",
    	help="path to output model")
    ap.add_argument("-pl", "--plot", default="data/plot.pdf",
    	help="path to output accuracy/loss plot")
    ap.add_argument("-n", "--new", action="store_true",
    	help="train a new model")
    ap.add_argument("-p2", "--phase-2", action="store_true",
    	help="traing phase 2 with direct SASA loss")
    args = vars(ap.parse_args())



    print("[INFO] training network...")

    if args["phase_2"]:
        #Not working. Might be impossible to write that kind of loss function 
        #because the input tensor has to be transformed to a np.array and back.
        #I don't think thats allowed
        with CustomObjectScope({'loss': mse_with_changable_weight(continuous_out_loss)}):
            old_model = load_model(args["model"])

        merge = concatenate(model.output)
        model = Model(inputs=old_model.input, outputs=merge)
        opt = Adam()


        with sqlite3.connect(database="/home/tim/Uni/BA/meta_material_databank/NN_smats.db") as conn:
            c = Crawler(directory="data/smat_data", cursor=conn.cursor())
        sli = fit.SingleLayerInterpolator(c)
        lb = fit.LabelBinarizer()

        batch_X = tf.tesor(shape=(BATCH_SIZE, NUMBER_OF_WAVLENGTHS, 2))
        callback = BatchUpdater(batch_X)

        model.compile(
            optimizer=opt,
            loss=SASA_loss(lb, c, sli, batch_X=batch_X) ,
            metrics=['accuracy']
            )


    #else do phase 1 training
    else:
        continuous_out_loss = tf.Variable(1/40000)
        callback = LossWeightsChanger(continuous_out_loss)

        if args["new"]:
            model = create_model()
            opt = Adam()#decay=INIT_LR / EPOCHS lr=INIT_LR,
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


    trainGen = batch_generator(args["batches"])
    validationGen = batch_generator(args["validation"])
    batch_count = len(os.listdir(f"{args['batches']}/X"))
    validation_count = len(os.listdir(f"{args['validation']}/X"))

    H = model.fit(
        trainGen,
	    steps_per_epoch=batch_count,
        validation_data=validationGen,
        validation_steps=validation_count,
        callbacks=[callback],
        epochs=EPOCHS,
        )

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    #set the matplotlib backend so figures can be saved in the background
    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    ax.minorticks_off()
    N = EPOCHS
    ax.plot(np.arange(0, N), H.history["discrete_out_loss"], label="train_loss", color="k")
    ax.plot(np.arange(0, N), H.history["discrete_out_accuracy"], label="train discrete acc", color="r")
    ax.plot(np.arange(0, N), H.history["continuous_out_accuracy"], label="train continuous acc", color="b")
    ax.plot(np.arange(0, N), H.history["val_discrete_out_accuracy"], label="validation discrete acc", color="r", linestyle="--")
    ax.plot(np.arange(0, N), H.history["val_continuous_out_accuracy"], label="validation continuous acc", color="b", linestyle="--")

    ax.set_title(f"Training Loss and Accuracy, LR: {INIT_LR}")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend(loc="upper left")
    fig.savefig(args["plot"])

    print("[DONE]")


#%%
