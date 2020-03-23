import os
import sys
import numpy as np
from scipy.optimize import minimize, curve_fit
import argparse
import pickle
import matplotlib.pyplot as plt
import sqlite3
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.losses import mean_squared_error

#Self written Modules
from sasa_db.crawler import Crawler
from sasa_phys.stack import *
from utils import LabelBinarizer, n_SiO2_formular, mean_squared_diff, Plotter, height_bound
from hyperparameters import *


OPTI_BOUND = [(0.6, 1.0)]

class SingleLayerInterpolator():
    """
    This class takes parameters of a single layer meta surface and
    looks into the database for similar layers which have been simulated. It then
    interpolates these to get an approximation for the behaviour of a layer
    with the provided parameters.

    # Arguments
        crawler: crawler obj.
        num_of_neigbours: int, how many similar layers should be
            considered for the interpolation
        power_faktor: int, exponent for inverse-distance-weight interpolation (IDW)


    """
    def __init__(self, crawler, num_of_neigbours=6, power_faktor=2):
        self.crawler = crawler
        self.num_of_neigbours = num_of_neigbours
        self.power_faktor = power_faktor
        self.interpolate = True


    def _set_grid_scale(self, param_dict):
        query=f"""SELECT
        MAX(wire.width) - MIN(wire.width),
        MAX(wire.length) - MIN(wire.length),
        MAX(wire.thickness) - MIN(wire.thickness),
        MAX(simulations.periode) - MIN(simulations.periode)
        FROM simulations
        INNER JOIN wire
        ON simulations.simulation_id = wire.simulation_id
        WHERE particle_material = '{param_dict["particle_material"]}'
        AND wire.hole = '{param_dict["hole"]}'"""

        self.crawler.cursor.execute(query)
        self.scale = np.array(self.crawler.cursor.fetchone())


    def _set_grid(self, param_dict):
        query=f"""SELECT wire.width, wire.length, wire.thickness,
        simulations.periode, simulations.simulation_id
        FROM simulations
        INNER JOIN wire
        ON simulations.simulation_id = wire.simulation_id
        WHERE particle_material = '{param_dict["particle_material"]}'
        AND wire.hole = '{param_dict["hole"]}'"""

        self.crawler.cursor.execute(query)
        data = np.array(self.crawler.cursor.fetchall())
        self.grid = data[:,:-1]
        self.ids = data[:,-1]
        self.grid = self.grid/self.scale


    def _sort_ids_and_distances(self, target):
        #scale target  [120, 250, ...] -> [0.8, 0.55, ...]
        target = target/self.scale
        #calculate distances d(target, grid)
        distances = np.sum((self.grid - target)**2, axis=1)
        #sort the ids by the distances
        idxs = distances.argsort()
        sorted_ids = self.ids[idxs]
        sorted_distances = distances[idxs]
        return sorted_ids, sorted_distances

    def closest_neigbor(self, param_dict):
        """
        Finds a pre calculated smat closest to the submitted parameters

        # Arguments
            param_dict: dict, contains all the parameters of a layer
            crawler: crawler obj

        # Returns
            smat: Lx4x4 Array

        """
        print("[INFO] closest_neigbor called")
        #pretty ridiculus query based on minimizing ABS(db_entry - target)
        query = f"""SELECT simulations.simulation_id
        FROM simulations
        INNER JOIN wire
        ON simulations.simulation_id = wire.simulation_id
        WHERE particle_material = '{param_dict["particle_material"]}'
        AND wire.hole = '{param_dict["hole"]}'
        ORDER BY ABS(wire.width - {param_dict["width"]})
        + ABS(wire.length - {param_dict["length"]})
        + ABS(wire.thickness - {param_dict["thickness"]})
        + ABS(simulations.periode - {param_dict["periode"]})
        LIMIT 1"""

        self.crawler.cursor.execute(query)
        id = self.crawler.cursor.fetchone()[0]
        smat = self.crawler.load_smat_by_id_npy(id)
        return smat


    def interpolate_smat(self, param_dict):
        self._set_grid_scale(param_dict)
        self._set_grid(param_dict)

        #print("[INFO] scale: ", self.scale)

        target = np.array([param_dict["width"], param_dict["length"], param_dict["thickness"], param_dict["periode"]])
        sorted_ids, sorted_distances = self._sort_ids_and_distances(target)
        #if the distance is close to 0 just return the closest neigbour
        if np.isclose(sorted_distances[0] + 1, 1):
            return self.closest_neigbor(param_dict)
        #calculate the weigths (IDW interpolation)
        weights = 1/sorted_distances[:self.num_of_neigbours]**self.power_faktor
        #scale weigths so sum(weights) = 1
        weights = weights/np.sum(weights)
        #calculate the interpolated smat
        interpolated_smat = np.zeros((NUMBER_OF_WAVLENGTHS,4,4), dtype=complex)
        for i in range(self.num_of_neigbours):
            id = sorted_ids[i]
            smat = self.crawler.load_smat_by_id_npy(id)

            if np.any(np.isnan(smat)):
                print(f"[WARNING] smat with ID: {id} contains nan's")
                continue

            interpolated_smat += weights[i]*smat

        #ensure there are no nan's in the interpolated_smat
        return interpolated_smat

def classify(model, spectrum, lb):
    #get the NN output
    discrete_out, continuous_out = model.predict(spectrum.reshape(1, NUMBER_OF_WAVLENGTHS, 2))

    #squeeze the additional dimension keras adds
    discrete_out = discrete_out[0]
    continuous_out = continuous_out[0]

    #classify it
    p1, p2, p_stack = classify_output(discrete_out, continuous_out, lb)

    return p1, p2, p_stack

def classify_output(discrete_out, continuous_out, lb):
    ##extract discrete parameters
    N = len(discrete_out)
    #round the prediction to ints: [0.2, 0.8] -> [0,1]
    enc_discrete1 = np.rint(discrete_out[:N//2])
    enc_discrete2 = np.rint(discrete_out[N//2:])

    params = lb.inverse_transform(np.array([enc_discrete1, enc_discrete2]))
    layer1 = params[0]
    layer2 = params[1]

    #fill the data into a dict example:
    #(Au, Holes) -> {particle_material : Au, hole: Holes}
    keys = list(MODEL_DISCRETE_PREDICTIONS)
    p1 = {keys[i] : layer1[i] for i in range(len(layer1))}
    p2 = {keys[i] : layer2[i] for i in range(len(layer2))}
    p_stack = {}

    #extract continuous parameters <- needs to be generalized
    p1["width"] = continuous_out[0]
    p1["length"] = continuous_out[1]
    p1["thickness"] = continuous_out[2]
    p1["periode"] = continuous_out[3]

    p2["width"] = continuous_out[4]
    p2["length"] = continuous_out[5]
    p2["thickness"] = continuous_out[6]
    p2["periode"] = continuous_out[7]

    p_stack["spacer_height"] = continuous_out[8]
    p_stack["angle"] = continuous_out[9]

    return p1, p2, p_stack



def param_dicts_to_arr(p1, p2, p_stack):
    """
    Turns parameter dictionaries into a numpy array

    # Arguments
        p1: dict, parameters of layer 1
        p2: dict, parameters of layer 2
        p_stack: dict, parameters of the stack

    # Returns
        array
    """
    return np.array([
            p1["width"],
            p1["length"],
            p1["thickness"],
            p1["periode"],
            p2["width"],
            p2["length"],
            p2["thickness"],
            p2["periode"],
            p_stack["angle"],
            p_stack["spacer_height"],
            ])

def param_dicts_update(p1, p2, p_stack, arr):

    p1["width"] = arr[0]
    p1["length"] = arr[1]
    p1["thickness"] = arr[2]
    p1["periode"] = arr[3]

    p2["width"] = arr[4]
    p2["length"] = arr[5]
    p2["thickness"] = arr[6]
    p2["periode"] = arr[7]

    p_stack["angle"] = arr[8]
    p_stack["spacer_height"] = arr[9]


def _outer_dist_to_bound(lower, upper, val):
    if val < lower:
        return (lower - val)/(upper - lower)
    elif val > upper:
        return (val - upper)/(upper - lower)
    else:
        return 0

def params_bounds_distance(p1, p2, p_stack, bounds):
    """
    Calculates total distance between the provided parameters and bounds.

    # Arguments
        p1: dict, layer 1 paramters
        p2: dict, layer 2 paramters
        p_stack: dict, stack paramters
        bounds: dict, parameter: [lower bound, upper bound]

    # Returns
        dist: float
    """
    dist = 0
    for key, bound in bounds.items():
        if key in p1:
            dist += _outer_dist_to_bound(bound[0], bound[1], p1[key])

        if key in p2:
            dist += _outer_dist_to_bound(bound[0], bound[1], p2[key])

        if key in p_stack:
            dist += _outer_dist_to_bound(bound[0], bound[1], p_stack[key])

    return dist

def calculate_spectrum(p1, p2, p_stack, c, sli):
    """
    Builds a SASA Stack with the provided parameters and calculates its
    spectrum

    # Arguments
        p1: dict, parameters of layer 1
        p2: dict, parameters of layer 2
        p_stack: dict, parameters of the stack
        c: Crawler object
        sil: SingleLayerInterpolator obj.

    # Returns
        spec: Lx2 array, stacked X and Y transmission spectra
    """
    if not sli.interpolate:
        smat1 = sli.closest_neigbor(p1)
        smat2 = sli.closest_neigbor(p2)
    else:
        smat1 = sli.interpolate_smat(p1)
        smat2 = sli.interpolate_smat(p2)

    wav = np.linspace(
        WAVLENGTH_START,
        WAVLENGTH_STOP,
        NUMBER_OF_WAVLENGTHS)

    SiO2 = n_SiO2_formular(wav)

    l1 = MetaLayer(smat1, SiO2, SiO2)
    l1.rotate(p_stack["angle"])
    l2 = MetaLayer(smat2, SiO2, SiO2)

    spacer = NonMetaLayer(SiO2, height=p_stack["spacer_height"])

    stack = Stack([l1, spacer, l2], wav, SiO2, SiO2)
    smat = stack.build()
    spec_x = np.abs(smat[:, 0, 0])**2 / SiO2
    spec_y = np.abs(smat[:, 1, 1])**2 / SiO2
    spec  = np.stack((spec_x, spec_y), axis=1)

    return spec

def set_defaults(p1, p2, p_stack):
    p1["width"] = 250
    p1["length"] = 250
    p1["thickness"] = 150
    p1["periode"] = 400

    p2["width"] = 250
    p2["length"] = 250
    p2["thickness"] = 150
    p2["periode"] = 400

    p_stack["angle"] = 0.0
    p_stack["spacer_height"] = 1.0


def loss(arr, target_spec, p1, p2, p_stack, bounds, crawler, plotter, sli, stp):
    """
    This loss function is minimized by the scipy optimizer. It takes all the
    parameters of a stack, calculates the resulting transmission spectrum and
    compares it to the target. Additionally it checks if physical bounds are
    violated and adds `params_bounds_distance()` to the loss value.

    # Arguments
        arr: array, the scipy optimizer needs the first argument to be an array
            with all the tuneable parameters.
        target_spec: Lx2 array
        p1: dict, parameters of layer 1
        p2: dict, parameters of layer 2
        p_stack: dict, parameters of the stack
        bounds: dict, {parameter: [lower bound, upper bound]}
        crawler: crawler object to access the db
        plotter: plotter object
        sli: SingleLayerInterpolator object

    # Returns
        loss_val: float

    """
    stp += 1

    param_dicts_update(p1, p2, p_stack, arr)

    current_spec = calculate_spectrum(p1, p2, p_stack, crawler, sli)
    loss_val = mean_squared_diff(current_spec, target_spec)

    #update the specer height bound
    d_min1 = height_bound(p1["periode"], WAVLENGTH_STOP)
    d_min2 = height_bound(p2["periode"], WAVLENGTH_STOP)
    d_min = min(d_min1, d_min2)
    bounds["spacer_height"][0] = d_min
    #print("[INFO] d_min...", d_min)

    #check if the parameters satisfy the bounds
    dist = params_bounds_distance(p1, p2, p_stack, bounds)
#    if dist != 0:
#        print(f"[INFO] Distance to bounds: {dist:.3f}")

    current_text = plotter.write_text(p1, p2, p_stack, loss_val)

    plotter.update(current_spec, target_spec, current_text)

    return loss_val + dist**3


#%%
if __name__ == '__main__':

    args = {"model" : "data/stacker.h5",
            "spectrum" : "test_spectrum.npy",
            "index" : 0,
            }

    #%% construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("spectrum", metavar="s",
        help="path to target spectrum .npy file")
    ap.add_argument("-m", "--model", default="stacker.h5",
    	help="path to trained model model")
    ap.add_argument("-db", "--database", default="data/NN_smats.db",
                        help="sqlite database containing the adresses")
    ap.add_argument("-S", "--smats", default="data/smats",
                        help="directory containing the smats for interpolation")
    ap.add_argument("-i", "--index", default=0, type=int)
    ap.add_argument("-I", "--interpolate", action="store_false", default=True)
    args = vars(ap.parse_args())
    #%%

    print("[INFO] loading network...")
    #the scope is nessecary beacuse I used a custom loss for training
    with CustomObjectScope({'loss':mean_squared_error}):
        model = load_model(args["model"])

    print("[INFO] loading input spectrum...")
    lb = LabelBinarizer()
    target_spectrum = np.load(args["spectrum"])[args['index']]

    with sqlite3.connect(database=args['database']) as conn:
        crawler = Crawler(directory=args['smats'], cursor=conn.cursor())


    #Phase 1: use the model to get an initial guess
    print("[INFO] classifying spectrum...")
    p1, p2, p_stack = classify(model, target_spectrum, lb)
    print(p1)
    print(p2)
    print(p_stack)
    guess = param_dicts_to_arr(p1, p2, p_stack)


    #Phase 2: change the continuous to minimize a loss function

    bounds = {
        "width" : [40, 350],
        "length" : [40, 350],
        "thickness" : [20, 80],
        "periode" : [250, 700],
        "angle" : [0, 90],
        "spacer_height" : [0,0.3],
    }

    plt.ion()
    plotter = Plotter(ax_num=3)
    sli = SingleLayerInterpolator(crawler)
    sli.interpolate = args["interpolate"]
    stp = 0

    def callback(xk):
        global stp
        print(stp)
        stp+=1

    print("[INFO] optimizing continuous parameters...")
    sol = minimize(
        loss, guess,
        args=(target_spectrum, p1, p2, p_stack, bounds, crawler, plotter, sli, stp),
        method="Nelder-Mead",
        callback=callback
    )
    print("[Done]")
    input()
