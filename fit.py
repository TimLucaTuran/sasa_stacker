#Fantasy code just to try the structure
import sys
sys.path.insert(0, '../SASA')

import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


#Self written Modules
from stack import *
from data_gen import create_random_stack, LabelBinarizer

def gen_single_layer(*params):
    pass

def mean_square_diff(current, target):
    """
    Calculates the mean squared diffrence between target and current

    Parameters
    ==========
    current : Lx4x4 array
        calculated array from SLN and SASA
    target  : Lx4x4 array
        goal array of optimation

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

    #round the prediction to ints: [0.2, 0.8] -> [0,1]
    enc_layer1 = np.rint(prob[:N//2])
    enc_layer2 = np.rint(prob[N//2:])

    params = lb.inverse_transform(np.array([enc_layer1, enc_layer2]))
    return params

def test(model, lb, data_directory):
    spectrum, p1, p2, params = create_random_stack(file_list, param_dict, data_directory)
    p = classify(model, spectrum, lb)

    print("Layer 1:", p1["particle_material"], p1["hole"], "Prediction:", p[0])
    print("Layer 2:", p2["particle_material"], p2["hole"], "Prediction:", p[1])
    fig, ax = plt.subplots()
    ax.plot(spectrum)






#%%
if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="data/stacker.model",
    	help="path to trained model model")
    ap.add_argument("-s", "--smat-directory", default="data/smat_data",
    	help="path to input directory containing .npy files")
    ap.add_argument("-p", "--params", default="data/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    args = vars(ap.parse_args())

    args = {"model" : "data/stacker.h5",
            "data_directory": "data/smat_data",
            "params": "data/params.pickle"}

    print("[INFO] loading network...")
    model = load_model(args["model"])


    file_list = os.listdir(args['data_directory'])
    with open(args["params"], "rb") as f:
        param_dict = pickle.load(f)
    spectrum, p1, p2, params = create_random_stack(file_list, param_dict, args["data_directory"])

    #Phase 1: use the model to classify the discrete parameters
    print("[INFO] classifying discrete parameters...")
    lb = LabelBinarizer()


    test(model,lb, args["data_directory"])

    #construct a stack with the recived discrete parameters
    #and set defaults for the continuous ones

    mat1 = gen_single_layer()
    layer1 = MetaLayer(mat1)

    mat2 = gen_single_layer()
    layer2 = MetaLayer(mat2)

    spacer = NonMetaLayer(SiO2, height=h)
    #Phase 2: change the continuous to minimize a loss function

    stack = Stack([layer1, spacer, layer2], wav, cladding, substrate)

    minimize_loss(mean_square_diff, target_spec, stack)
