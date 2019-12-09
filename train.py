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
from tensorflow.keras.layers import Input, Dense, MaxPooling1D, Dropout, Conv1D, GlobalAveragePooling1D, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
#Self written modules
from crawler import Crawler
from stack import *

MODEL_INPUTS = 128
MODEL_DISCRETE_OUTPUTS = 8
MODEL_CONTINUOUS_OUTPUTS = 8
MODEL_DISCRETE_PREDICTIONS = {
    "particle_material" : ["Au", "Al"],
    "hole" : ["holes", "no holes"]
    }

BATCH_SIZE = 128
EPOCHS = 10
INIT_LR = 1e-3

#%%

def create_model():
    inp = Input(shape=(MODEL_INPUTS))
    x = Reshape((MODEL_INPUTS, 1)) (inp)
    x = Conv1D(64, 10, activation='relu')(x)
    x = Conv1D(64, 10, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 10, activation='relu')(x)
    conv_out = Conv1D(128, 10, activation='relu')(x)
    #discrete branch
    x = GlobalAveragePooling1D()(conv_out)
    x = Dropout(0.5)(x)
    discrete_out = Dense(MODEL_DISCRETE_OUTPUTS, activation='sigmoid', name='discrete_out')(x)
    #continuous branch
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    continuous_out = Dense(MODEL_CONTINUOUS_OUTPUTS, activation='linear', name='continuous_out')(x)

    model = Model(inputs=inp, outputs=[discrete_out, continuous_out])
    return model

def batch_generator(batch_dir):
    """
    Just load the batches created by data_gen.py

    """

    inp_batches = []
    while True:
        #reset x_batches once are batches are used up
        if len(inp_batches) == 0:
            inp_batches = os.listdir(f"{batch_dir}/input")

        idx = random.randint(0, len(inp_batches)-1)
        batch = inp_batches[idx][:-4]#[:-4] to remove the .npy

        x = np.load(f"{batch_dir}/input/{batch}.npy")
        discrete_out = np.load(f"{batch_dir}/discrete_out/{batch}.npy")

        with open(f"{batch_dir}/params/{batch}.pickle", "rb") as f:
            params = pickle.load(f)
        continuous_out = np.zeros((BATCH_SIZE, MODEL_CONTINUOUS_OUTPUTS))

        for i in range(BATCH_SIZE): #needs to be generalized
            layer1, layer2, stack = params[i]

            continuous_out[i,0] = layer1["width"]
            continuous_out[i,1] = layer1["thickness"]
            continuous_out[i,2] = layer1["periode"]

            continuous_out[i,3] = layer2["width"]
            continuous_out[i,4] = layer2["thickness"]
            continuous_out[i,5] = layer2["periode"]

            continuous_out[i,6] = stack["spacer_height"]
            continuous_out[i,7] = stack["angle"]

        del inp_batches[idx]

        yield (x, [discrete_out, continuous_out])




#%%
if __name__ == '__main__':

    #%% construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batches", default="data/batches",
    	help="path to directory containing the training batches")
    ap.add_argument("-p", "--params", default="data/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    ap.add_argument("-m", "--model", default="data/stacker.h5",
    	help="path to output model")
    ap.add_argument("-pl", "--plot", default="data/plot.png",
    	help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())

    #set the matplotlib backend so figures can be saved in the background
    matplotlib.use("Agg")

    print("[INFO] training network...")
    model = create_model()
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    losses = {
        'discrete_out' : 'binary_crossentropy',
        'continuous_out' : 'mse',
        }
    model.compile(optimizer=opt, loss=losses, metrics=['accuracy'])



    trainGen = batch_generator(args["batches"])
    validationGen = batch_generator("data/validation")
    batch_count = len(os.listdir(f"{args['batches']}/input"))
    validation_count = len(os.listdir("data/validation/input"))

    H = model.fit_generator(
        trainGen,
	    steps_per_epoch=batch_count,
        validation_data=validationGen,
        validation_steps=validation_count,
        epochs=EPOCHS,
        use_multiprocessing=True)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    print(H.history)


    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["discrete_out_loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["discrete_out_accuracy"], label="train discrete acc")
    plt.plot(np.arange(0, N), H.history["continuous_out_accuracy"], label="train continuous acc")
    plt.plot(np.arange(0, N), H.history["val_discrete_out_accuracy"], label="validation discrete acc")
    plt.plot(np.arange(0, N), H.history["val_continuous_out_accuracy"], label="validation continuous acc")

    plt.title(f"Training Loss and Accuracy, LR: {INIT_LR}")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args["plot"])

    print("[DONE]")


#%%
