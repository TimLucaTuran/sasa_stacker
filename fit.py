#Fantasy code just to try the structure
import sys
sys.path.insert(0, '../SASA')

import os
import numpy as np
import argparse
import pickle
import tensorflow as tf


#Self written Modules
from stack import *
from train import create_random_stack

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


def classify(model, spectrum):
    proba = model.predict(spectrum)[0]
    return proba
model = tf.keras.models.load_model("data/stacker.h5")
spectrum = np.expand_dims(np.random.randn(128), axis=0)

classify(model, spectrum)

#%%
if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="data/stacker.model",
    	help="path to trained model model")
    ap.add_argument("-l", "--labelbin", default="data/mlb.pickle",
    	help="path to label binarizer")
    args = vars(ap.parse_args())

    #args = {"model" : "data/stacker.model"}

    print("[INFO] loading network...")
    model = load_model(args["model"])
    mlb = pickle.loads(open(args["labelbin"], "rb").read())


    file_list = os.listdir(args['data_directory'])
    with open(args["params"], "rb") as f:
        param_dict = pickle.load(f)
        spectrum, p1, p2, params = create_random_stack(file_list, param_dict)

    #Phase 1: use the model to classify the discrete parameters
    print("[INFO] classifying discrete parameters...")
    prob = model.predict(np.expand_dims(spectrum, axis=0))[0]
    print(prob)
    idxs = np.argsort(prob)[::-1][:2]
    print(idxs)
    #construct a stack with the recived discrete parameters
    #and set defaults for the continuous ones
    sys.exit()
    mat1 = gen_single_layer()
    layer1 = MetaLayer(mat1)

    mat2 = gen_single_layer()
    layer2 = MetaLayer(mat2)

    spacer = NonMetaLayer(SiO2, height=h)

    stack = Stack([layer1, spacer, layer2], wav, cladding, substrate)

    #Phase 2: change the continuous to minimize a loss function
    minimize_loss(mean_square_diff, target_spec, stack)
