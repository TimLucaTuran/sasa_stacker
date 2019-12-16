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
import time



#Self written Modules
from stack import *
from data_gen import create_random_stack, LabelBinarizer, n_SiO2_formular
from crawler import Crawler
import data_gen, fit, train
from fit import Plotter

def test(model, lb, spec_name, spec_num=0):
    spectrum = np.load(f"data/batches/X/{spec_name}")[spec_num]
    p1_pred, p2_pred = classify(model, spectrum, lb)

    print("Layer 1:", p1["particle_material"], p1["hole"],"\nPrediction:", p1_pred)
    print("Layer 2:", p2["particle_material"], p2["hole"],"\nPrediction:", p2_pred)
    fig, ax = plt.subplots()
    ax.plot(spectrum)

    return fig, ax

def NN_test_loop(crawler, lb):

    while True:
        spectrum, true1, true2, true_stack = data_gen.create_random_stack(
            crawler, param_dict)

        l1 , l2, stack = fit.classify(model, spectrum, lb)

        plotter = Plotter(ax3_on=True)
        pred_text = plotter.write_text(l1, l2, stack, loss_val=0)
        true_text = plotter.write_text(true1, true2, true_stack, loss_val=0)
        plotter.double_text(spectrum, pred_text, true_text)
        plt.show()




def show_stack_info():
    p = Plotter()
    #load spectrum
    spec = np.load(args['stack'])[args['index']]

    #load stack parameters
    name = args['stack'].split("/")[-1][:-4]
    with open(f"{args['batch_dir']}/params/{name}.pickle", "rb") as f:
        stack_params = pickle.load(f)

    p1, p2, p_stack = stack_params[args['index']]

    text = p.write_text(p1, p2, p_stack, loss_val=0)
    p.update(spec, spec, text)
    plt.show()

#%%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--stack", required=False)
    ap.add_argument("-i", "--index", default=0, type=int)
    ap.add_argument("-b", "--batch-dir", default="data/batches")
    ap.add_argument("-l", "--loop", action="store_true", help="looping NN predictions")
    args = vars(ap.parse_args())

    model = load_model("data/stacker.h5")
    lb = data_gen.LabelBinarizer()
    file_list = os.listdir("data/smat_data")
    with open("data/params.pickle", "rb") as f:
        param_dict = pickle.load(f)

    conn = sqlite3.connect("../meta_material_databank/NN_smats.db")
    c = Crawler(directory="data/smat_data", cursor=conn.cursor())


    if args["stack"] is not None:
        show_stack_info()

    if args["loop"]:
        NN_test_loop(c, lb)
