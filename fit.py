#Fantasy code just to try the structure
import sys
sys.path.insert(0, '../SASA')
sys.path.insert(0, "../meta_material_databank")

import os
import numpy as np
from scipy.optimize import minimize
import argparse
import pickle
import matplotlib.pyplot as plt
import sqlite3
from tensorflow.keras.models import load_model


#Self written Modules
from stack import *
from data_gen import create_random_stack, LabelBinarizer, n_SiO2_formular
from crawler import Crawler
import train

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
    prob = model.predict(spectrum.reshape(1,128))[0]
    N = len(prob)


    #round the prediction to ints: [0.2, 0.8] -> [0,1]
    enc_layer1 = np.rint(prob[:N//2])
    enc_layer2 = np.rint(prob[N//2:])

    params = lb.inverse_transform(np.array([enc_layer1, enc_layer2]))
    layer1 = params[0]
    layer2 = params[1]

    #fill the data into a dict example:
    #(Au, Holes) -> {particle_material : Au, hole: Holes}
    keys = list(train.MODEL_PREDICTIONS)
    p1 = {keys[i] : layer1[i] for i in range(len(layer1))}
    p2 = {keys[i] : layer2[i] for i in range(len(layer2))}
    return p1, p2, prob


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

def param_dicts_update(p1, p2, p_stack, arr):

    p1["width"] = arr[0]
    p1["thickness"] = arr[1]
    p1["periode"] = arr[2]

    p2["width"] = arr[3]
    p2["thickness"] = arr[4]
    p2["periode"] = arr[5]

    p_stack["angle"] = arr[6]
    p_stack["spacer_height"] = arr[7]


def calculate_spectrum(p1, p2, p_stack, c):
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
    smat = stack.build()
    spectrum = np.abs( smat[:, 2, 2] )**2 / SiO2

    return spectrum

def set_defaults(p1, p2, p_stack):
    p1["width"] = 250
    p1["thickness"] = 150
    p1["periode"] = 400

    p2["width"] = 250
    p2["thickness"] = 150
    p2["periode"] = 400

    p_stack["angle"] = 0.0
    p_stack["spacer_height"] = 1.0

def loss(arr, target_spec, p1, p2, p_stack, crawler, plotter):
    param_dicts_update(p1, p2, p_stack, arr)

    current_spec = calculate_spectrum(p1, p2, p_stack, crawler)
    current_text = plotter.write_text(p1, p2, p_stack)

    plotter.update(current_spec, target_spec, current_text)

    loss_val = mean_square_diff(current_spec, target_spec)

    print("w1: {:.0f} t1: {:.0f} p1: {:.0f} w2: {:.0f} t2: {:.0f} p2: {:.0f} h: {:.2f} a: {:.0f}".format(
        p1["width"], p1["thickness"], p1["periode"], p2["width"], p2["thickness"], p2["periode"], p_stack["spacer_height"], p_stack["angle"]
    ))
    return loss_val

class Plotter():
    #Suppose we know the x range
    min_x = 0
    max_x = 128

    def __init__(self):
        #Set up plot
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2)
            #Autoscale on unknown axis and known lims on the other
        self.ax1.set_autoscaley_on(True)
        self.ax1.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax1.grid()

    def write_text(self, p1, p2, p_stack):
        text = f"""
Layer 1:
material: {p1['particle_material']}
holes: {p1['hole']}
width: {p1['width']:.0f}
thickness: {p1['thickness']:.0f}
periode: {p1['periode']:.0f}

Layer 2:
material: {p2['particle_material']}
holes: {p2['hole']}
width: {p2['width']:.0f}
thickness: {p2['thickness']:.0f}
periode: {p2['periode']:.0f}

Stack
spacer_height: {p_stack['spacer_height']:.2f}
angle: {p_stack['angle']:.0f}
"""
        return text

    def update(self, current_spec, target_spec, text):
        #Update data (with the new _and_ the old points)
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.plot(target_spec)
        self.ax1.plot(current_spec)
        self.ax2.text(0.1, 0.1, text)

        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
#%%
if __name__ == '__main__':

    args = {"model" : "data/stacker.h5",
            "spectrum" : "test_spectrum.npy"}

    #%% construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="data/stacker.h5",
    	help="path to trained model model")
    ap.add_argument("-s", "--spectrum", required=True,
        help="path to target spectrum .npy file")
    ap.add_argument("-i", "--index", default=0, type=int)
    args = vars(ap.parse_args())
    #%%

    print("[INFO] loading network...")
    model = load_model(args["model"])

    print("[INFO] loading data...")
    lb = LabelBinarizer()
    target_spectrum = np.load(args["spectrum"])[args['index']]

    with sqlite3.connect(database="/home/tim/Uni/BA/meta_material_databank/NN_smats.db") as conn:
        c = Crawler(directory="data/smat_data", cursor=conn.cursor())


    #Phase 1: use the model to classify the discrete parameters
    print("[INFO] classifying discrete parameters...")

    p1, p2, _ = classify(model, target_spectrum, lb)
    p_stack = {}
    set_defaults(p1, p2, p_stack)
    #construct a stack with the recived discrete parameters
    #and set defaults for the continuous ones

    #Phase 2: change the continuous to minimize a loss function
    guess = param_dicts_to_arr(p1, p2, p_stack)

    b_width = (50.0, 500.0)
    b_thick = (10.0, 150.0)
    b_periode = (100.0, 725.0)
    b_angle = (0.0, 90.0)
    b_heigth = (0.0, 2.0)
    bnds = (b_width, b_thick, b_periode,
            b_width, b_thick, b_periode,
            b_angle, b_heigth)

    plt.ion()
    plotter = Plotter()
    sol = minimize(
        loss, guess,
        args=(target_spectrum, p1, p2, p_stack, c, plotter),
        method="Nelder-Mead",
        bounds=bnds
    )
