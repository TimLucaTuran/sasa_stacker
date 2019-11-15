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
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
#Self written modules
from crawler import Crawler
from stack import *

MODEL_INPUTS = 128
MODEL_OUTPUTS = 8
BATCH_SIZE = 128
EPOCHS = 1


#%% construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--smat-directory", required=True,
	help="path to input directory containing .npy files")
ap.add_argument("-p", "--params", required=True,
	help="path to the .pickle file containing the smat parameters")
ap.add_argument("-m", "--model", default="data/stacker.model",
	help="path to output model")
ap.add_argument("-l", "--labelbin", default="data/mlb.pickle",
	help="path to output label binarizer")
ap.add_argument("-pl", "--plot", default="data/plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

#%%
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
    m1 =  np.load("{}/{}".format(args['smat_directory'], file1))
    #load smat2
    file2 = random.choice(file_list)
    p2 = param_dict[file2]
    m2 =  np.load("{}/{}".format(args['smat_directory'], file2))


    wav = np.linspace(p1['wavelength_start'],
                      p1['wavelength_stop'], p1['spectral_points'])
    SiO2 = n_SiO2_formular(wav)

    l1, l2 = MetaLayer(m1, SiO2, SiO2), MetaLayer(m2, SiO2, SiO2)

    phi = random.uniform(0,90)
    l1.rotate(phi)

    h = random.uniform(0.1, 0.3)
    spacer = NonMetaLayer(SiO2, height=h)

    s = Stack([l1, spacer, l2], wav, SiO2, SiO2)
    smat = s.build()
    spectrum = np.abs( smat[:, 2, 2] )**2 / SiO2

    p_stack = { 'phi' : phi,
               'height': h,
             }
    return spectrum, p1, p2, p_stack

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

def batch_generator(size, mlb, file_list, param_dict):
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

    #Infinite loop, yields one batch per itteration
    while True:
        model_in = np.zeros((size, MODEL_INPUTS))
        labels1 = []
        labels2 = []

        for i in range(size):
            #generate stacks until one doesn't block all incomming light
            while True:
                spectrum, p1, p2, _ = create_random_stack(file_list, param_dict)
                if np.max(spectrum) > 0.1:
                    break

            model_in[i] = spectrum
            labels1.append((p1['particle_material'].strip(), p1['hole']),)
            labels2.append((p2['particle_material'].strip(), p2['hole']),)

        #encode the labels
        enc1 = mlb.fit_transform(labels1)
        enc2 = mlb.fit_transform(labels2)

        model_out = np.concatenate((enc1, enc2), axis=1)

        yield (model_in, model_out)


#%%
if __name__ == '__main__':
    #args = {'smat_directory': "data/smat_data",
    #        "params" : "data/params.pickle"}

    discrete_params = ['Au', 'Al', 'holes', 'no holes']
    mlb = MultiLabelBinarizer(classes=np.array(discrete_params, dtype=object))

    print("[INFO] loading data...")
    file_list = os.listdir(args['smat_directory'])
    with open(args["params"], "rb") as f:
        param_dict = pickle.load(f)


    print("[INFO] training network...")
    model = create_model()
    trainGen = batch_generator(BATCH_SIZE, mlb, file_list, param_dict)

    H = model.fit_generator(
        trainGen,
	    steps_per_epoch=10,
        validation_data=trainGen,
        validation_steps=1,
        epochs=EPOCHS,
        use_multiprocessing=True)

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

    print("[DONE]")


#%%
