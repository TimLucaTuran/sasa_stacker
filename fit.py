#Fantasy code just to try the structure
import sys
sys.path.insert(0, '../SASA')
sys.path.insert(0, "../meta_material_databank")

import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import sqlite3
from tensorflow.keras.models import load_model


#Self written Modules
from stack import *
from data_gen import create_random_stack, LabelBinarizer
from crawler import Crawler
import train

conn = sqlite3.connect(database="/home/tim/Uni/BA/meta_material_databank/NN_smats.db")
c = Crawler(directory="data/smat_data", cursor=conn.cursor())
param_dict = c.extract_params(1000)
param_dict

def single_layer_lookup(param_dict, crawler):
    """
    Finds a pre calculated smat closest to the submitted parameters

    Parameters
    ----------
    param_dict : dict
        contains all the parameters of a layer
    crawler : crawler obj

    Returns
    -------
    smat : Lx4x4 Array

    """
    #pretty ridiculus query based on minimizing ABS(db_entry - target)
    query = f"""SELECT simulations.simulation_id
    FROM simulations
    INNER JOIN square
    ON simulations.simulation_id = square.simulation_id
    WHERE particle_material = '{param_dict["particle_material"]}'
    AND square.hole = '{param_dict["hole"]}'
    ORDER BY ABS(square.width - {param_dict["width"]})
    + ABS(square.thickness - {param_dict["thickness"]})
    + ABS(simulations.periode - {param_dict["periode"]})
    LIMIT 1"""

    crawler.cursor.execute(query)
    id = crawler.cursor.fetchone()[0]
    smat = crawler.load_smat_by_id_npy(id)
    return smat



def mean_square_diff(current, target):
    """
    Calculates the mean squared diffrence between target and current smat

    Parameters
    ==========
    current : Lx4x4 array
        calculated smat from SASA
    target  : Lx4x4 array
        target smat of optimation

    Returns
    =======
    output : float
        real error value
    """
    return np.sum(np.abs(current - target)**2)

def minimize_loss(loss, target, stack):

    #define bounds
    b_width = (50.0, 500.0)
    b_thick = (10.0, 150.0)
    b_periode = (100.0, 725.0)
    b_angle = (0.0, 90.0)
    b_heigth = (0.0, 2.0)
    bnds = (b_width, b_thick, b_periode,
            b_width, b_thick, b_periode,
            b_heigth)


def classify(model, spectrum, lb):
    prob = model.predict(spectrum.reshape(1,128,1))[0]
    N = len(prob)
    print("NN Out: ", prob)

    #round the prediction to ints: [0.2, 0.8] -> [0,1]
    enc_layer1 = np.rint(prob[:N//2])
    enc_layer2 = np.rint(prob[N//2:])

    params = lb.inverse_transform(np.array([enc_layer1, enc_layer2]))
    layer1 = params[0]
    layer2 = params[1]

    #fill the data into a dict example:
    #(Au, Holes) -> {particle_material : Au, hole: Holes}
    p1 = {train.MODEL_PREDICTIONS[i] : layer1[i] for i in range(len(layer1))}
    p2 = {train.MODEL_PREDICTIONS[i] : layer2[i] for i in range(len(layer2))}
    return p1, p2

def test(model, lb, data_directory):
    spectrum, p1, p2, params = create_random_stack(file_list, param_dict, data_directory)
    p1_pred, p2_pred = classify(model, spectrum, lb)

    print("Layer 1:", p1["particle_material"], p1["hole"],"\nPrediction:", p1_pred)
    print("Layer 2:", p2["particle_material"], p2["hole"],"\nPrediction:", p2_pred)
    fig, ax = plt.subplots()
    ax.plot(spectrum)

    return fig, ax

def param_dicts_to_arr(p1, p2, p_stack):
    """
    Turns parameter dictionaries into a numpy array

    Parameters
    ----------
    p1 : dict
        parameters of layer 1
    p2 : dict
        parameters of layer 2
    p_stack : dict
        parameters of the stack

    Returns
    -------
    array
    """
    return np.array([
            p1["width"],
            p1["thickness"],
            p1["periode"],
            p2["width"],
            p2["thickness"],
            p2["periode"],
            p_stack["angle"],
            p_stack["spacer_height"],
            ])

def create_stack(p1, p2, p_stack, c):
    """
    Builds a SASA Stack with the provided parameters

    Parameters
    ----------
    p1 : dict
        parameters of layer 1
    p2 : dict
        parameters of layer 2
    p_stack : dict
        parameters of the stack
    c : crawler object

    Returns
    -------
    stack : SASA Stack object
    """
    smat1 = single_layer_lookup(p1, c)
    smat2 = single_layer_lookup(p2, c)

    wav = np.linspace(0.5, 1, 128)
    SiO2 = n_SiO2_formular(wav)

    l1 = MetaLayer(smat1, SiO2, SiO2)
    l1.rotate(p_stack["angle"])
    l2 = MetaLayer(smat2, SiO2, SiO2)

    spacer = NonMetaLayer(SiO2, height=p_stack["spacer_height"])

    stack = Stack([l1, spacer, l2], wav, SiO2, SiO2)
    return stack

def set_defaults(p1, p2, p_stack):
    p1["width"] = 250
    p1["thickness"] = 150
    p1["periode"] = 400

    p2["width"] = 250
    p2["thickness"] = 150
    p2["periode"] = 400

    p_stack["angle"] = 0.0
    p_stack["spacer_height"] = 1.0


#%%
if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="data/stacker.model",
    	help="path to trained model model")
    ap.add_argument("-s", "--spectrum", required=true,
        help="path to target spectrum .npy file")
    args = vars(ap.parse_args())
    #%%
    args = {"model" : "data/stacker.h5",
            "data_directory": "data/smat_data",
            "params": "data/params.pickle"}

    print("[INFO] loading network...")
    model = load_model(args["model"])
    lb = LabelBinarizer()
    spectrum = np.load(args["spectrum"])


    #Phase 1: use the model to classify the discrete parameters
    print("[INFO] classifying discrete parameters...")

    p1, p2 = classify(model, spectrum, lb)
    p_stack = {}
    set_defaults(p1, p2, p_stack)
    #construct a stack with the recived discrete parameters
    #and set defaults for the continuous ones
    stack = create_stack(p1, p2, p_stack)


    #Phase 2: change the continuous to minimize a loss function



    minimize_loss(mean_square_diff, target_spec, stack)
