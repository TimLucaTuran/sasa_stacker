import sys
sys.path.insert(0, "../meta_material_databank")
sys.path.insert(0, "../SASA")

from datetime import datetime
import random
import os
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
#self written modules
from crawler import Crawler
from stack import *
import train



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


def create_random_stack(file_list, param_dict, smat_directory):
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
    m1 =  np.load("{}/{}".format(smat_directory, file1))
    #load smat2
    file2 = random.choice(file_list)
    p2 = param_dict[file2]
    m2 =  np.load("{}/{}".format(smat_directory, file2))


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

    p_stack = { 'angle' : phi,
               'spacer_height': h,
             }
    return spectrum, p1, p2, p_stack


def create_batch(size, mlb, file_list, param_dict):
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

    model_in = np.zeros((size, train.MODEL_INPUTS))
    labels1 = []
    labels2 = []

    for i in range(size):
        #generate stacks until one doesn't block all incomming light
        while True:
            spectrum, p1, p2, _ = create_random_stack(file_list, param_dict, args["smat_directory"])
            if np.max(spectrum) > 0.1:
                break

        #save the input spectrum
        model_in[i] = spectrum

        #save the layer parameters which led to the spectrum
        label1 = [p1[key].strip() for key in train.MODEL_PREDICTIONS]
        label2 = [p2[key].strip() for key in train.MODEL_PREDICTIONS]

        labels1.append(label1)
        labels2.append(label2)

    #encode the labels
    enc1 = mlb.fit_transform(labels1)
    enc2 = mlb.fit_transform(labels2)

    model_out = np.concatenate((enc1, enc2), axis=1)

    return (model_in, model_out)

def LabelBinarizer():
    discrete_params = ['Au', 'Al', 'holes', 'no holes']
    mlb = MultiLabelBinarizer(classes=np.array(discrete_params, dtype=object))
    mlb.fit_transform([['Au', 'holes']])
    return mlb

#%%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--smat-directory", default="data/smat_data",
    	help="path to input directory containing .npy files")
    ap.add_argument("-p", "--params", default="data/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    ap.add_argument("-n", "--number-of-batches", default=10, type=int)
    args = vars(ap.parse_args())



    print("[INFO] loading data...")
    lb = LabelBinarizer()
    file_list = os.listdir(args['smat_directory'])
    with open(args["params"], "rb") as f:
        param_dict = pickle.load(f)


    for i in range(args["number_of_batches"]):
        print("[INFO] creating batch ", i+1)
        x, y = create_batch(train.BATCH_SIZE, lb, file_list, param_dict)
        ts = str(datetime.now()).replace(" ", "_")
        np.save("data/batches/X/{}.npy".format(ts), x)
        np.save("data/batches/Y/{}.npy".format(ts), y)
