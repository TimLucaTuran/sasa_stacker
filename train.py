import sys
sys.path.insert(0, "../meta_material_databank")
sys.path.insert(0, "../SASA")
#set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

#standard library modules
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import argparse
import pickle
import cProfile
#NN modules
from tensorflow.keras.layers import Input, Dense, MaxPooling1D, Dropout, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
#Self written modules
from crawler import Crawler
from stack import *

MODEL_INPUTS = 128
MODEL_OUTPUTS = 8
BATCH_SIZE = 128
EPOCHS = 10

#%%

def create_model():
    inp = Input(shape=(MODEL_INPUTS, 1))
    x = Conv1D(64, 3, activation='relu')(inp)
    x = Conv1D(64, 3, activation='relu')(x)
    #x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    out = Dense(MODEL_OUTPUTS, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def batch_generator():
    """
    Just load batches created by data_gen.py

    """
    while True:
        x_batches = os.listdir("data/batches/X")
        batch = random.choice(x_batches)

        x = np.load("data/batches/X/{}".format(batch)).reshape(BATCH_SIZE, MODEL_INPUTS, 1)
        y = np.load("data/batches/Y/{}".format(batch))
        yield (x, y)




#%%
if __name__ == '__main__':
    #%% construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--smat-directory", default="data/smat_data",
    	help="path to input directory containing .npy files")
    ap.add_argument("-p", "--params", default="data/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    ap.add_argument("-m", "--model", default="data/stacker.h5",
    	help="path to output model")
    ap.add_argument("-l", "--labelbin", default="data/mlb.pickle",
    	help="path to output label binarizer")
    ap.add_argument("-pl", "--plot", default="data/plot.png",
    	help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    #args = {'smat_directory': "data/smat_data",
    #        "params" : "data/params.pickle"}

    print("[INFO] training network...")
    model = create_model()
    trainGen = batch_generator()

    H = model.fit_generator(
        trainGen,
	    steps_per_epoch=BATCH_SIZE,
        validation_data=trainGen,
        validation_steps=BATCH_SIZE//10,
        epochs=EPOCHS,
        use_multiprocessing=True)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    """# save the multi-label binarizer to disk
    print("[INFO] serializing label binarizer...")
    with open(args["labelbin"], "wb") as f:
        f.write(pickle.dumps(mlb))"""

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args["plot"])

    print("[DONE]")


#%%
