#set the matplotlib backend so figures can be saved in the background
import sys
sys.path.insert(0, "../meta_material_databank")
sys.path.insert(0, "../SASA")
import matplotlib
matplotlib.use("Agg")

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import argparse
import pickle
#NN Modules
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
#Self written Modules
from crawler import Crawler
from stack import *

MODEL_INPUTS = 128
MODEL_OUTPUTS = 8
BATCH_SIZE = 512
EPOCHS = 5


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data-directory", required=True,
	help="path to input directory containing .npy files")
ap.add_argument("-pa", "--params", required=True,
	help="path to params pickle containing the smat parameters")
ap.add_argument("-m", "--model", default="stacker.model",
	help="path to output model")
ap.add_argument("-l", "--labelbin", default="mlb.pickle",
	help="path to output label binarizer")
ap.add_argument("-pl", "--plot", default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


def n_SiO2_formular(w):
    """
    Calculates the refractiv index of SiO2

    Parameters
    ----------
    w : vec
        wavelengths in micro meters

    Returns
    -------
    n : vec
        refractiv indeces
    """
    a1 = 0.6961663
    a2 = 0.4079426
    a3 = 0.8974794
    c1 = 0.0684043
    c2 = 0.1162414
    c3 = 9.896161
    n = np.sqrt(a1*w**2/(w**2 - c1**2) +
        a2*w**2/(w**2 - c2**2) + a3*w**2/(w**2 - c3**2) + 1)
    return n

def create_random_stack(file_list, param_dict):
    """
    Generates a random 2-Layer Stack and returns it's spectrum calculated via
	SASA and the generated parameters

    Parameters
    ----------
    samt1 : str
    smat2 : str
        these need to have the same
        wavelength_start/stop and spectral_points
	crawler : Crawler object

    Returns
    -------
    spectrum : array
    p1 : dict
        layer 1 parameters
    p2 : dict
        layer 2 parameters
    params : dict
        stack parameters

    """
    #load smat1
    file1 = random.choice(file_list)
    p1 = param_dict[file1]
    m1 =  np.load("{}/{}".format(args['data_directory'], file1))
    #load smat2
    file2 = random.choice(file_list)
    p2 = param_dict[file2]
    m2 =  np.load("{}/{}".format(args['data_directory'], file2))


    wav = np.linspace(p1['wavelength_start'],
                      p1['wavelength_stop'], p1['spectral_points'])
    SiO2 = n_SiO2_formular(wav)

    l1, l2 = MetaLayer(m1, SiO2, SiO2), MetaLayer(m2, SiO2, SiO2)

    phi = random.uniform(0,90)
    l1.rotate(phi)

    h = random.uniform(0.1, 0.3)
    spacer = NonMetaLayer(SiO2, height=h)

    s = Stack([l1, spacer, l2], wav, SiO2, SiO2)
    res = s.build()
    spectrum = np.abs( res[:, 2, 2] )**2 / SiO2

    params = { 'phi' : phi,
               'height': h,
             }
    return spectrum, p1, p2, params

def create_model():
    inp = Input(shape=(MODEL_INPUTS,))
    x = Dense(128, activation='relu')(inp)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    out = Dense(MODEL_OUTPUTS, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_minibatch(size, mlb, file_list, param_dict):
    """Uses create_random_stack() to create a minibatch

    Parameters
    ----------
    size : int
           the batch size
    ids : list
          all these need to have the same
          wavelength_start/stop and spectral_points
    crawler : Crawler obj
    mlb : MultiLabelBinarizer obj
          initialized to the discrete labels

    Returns
    -------
    model_in : size x MODEL_INPUTS Array
    model_out : size x MODEL_OUTPUTS Array

    """
    model_in = np.zeros((size, MODEL_INPUTS))
    model_out = np.zeros((size, MODEL_OUTPUTS))
    labels1 = []
    labels2 = []

    for i in range(size):
        while True:
            spec, p1, p2, params = create_random_stack(file_list, param_dict)
            if np.max(spec) > 0.1:
                break

        model_in[i] = spec
        labels1.append((p1['particle_material'], p1['hole']),)
        labels2.append((p2['particle_material'], p2['hole']),)

    #encode the labels
    enc1 = mlb.fit_transform(labels1)
    enc2 = mlb.fit_transform(labels2)

    model_out = np.concatenate((enc1, enc2), axis=1)

    return model_in, model_out



#%%
if __name__ == '__main__':

    discrete_params = ['Au', 'Al', 'hole', 'no hole']
    mlb = MultiLabelBinarizer(classes=np.array(discrete_params, dtype=object))

    print("[INFO] loading data...")
    file_list = os.listdir(args['data_directory'])
    with open(args["params"], "rb") as f:
        param_dict = pickle.load(f)
        print("[INFO] generating minibatch...")
        model_in, model_out = create_minibatch(BATCH_SIZE, mlb, file_list, param_dict)

    (trainX, testX, trainY, testY) = train_test_split(model_in, model_out,
                                                      test_size=0.1)

    print("[INFO] training network...")
    model = create_model()
    H = model.fit(model_in, model_out, epochs=EPOCHS, validation_data=(testX, testY))

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    # save the multi-label binarizer to disk
    print("[INFO] serializing label binarizer...")
    with open(args["labelbin"], "wb") as f:
        f.write(pickle.dumps(mlb))

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

    print("[INFO] done")


#%%
