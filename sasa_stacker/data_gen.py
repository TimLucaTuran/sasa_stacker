from datetime import datetime
import random
import os
import pickle
import argparse
import numpy as np
import sqlite3
from sklearn.preprocessing import MultiLabelBinarizer
#self written modules
from sasa_db.crawler import Crawler
from sasa_phys.stack import *
from train import NUMBER_OF_WAVLENGTHS, WAVLENGTH_START, WAVLENGTH_STOP, MODEL_DISCRETE_PREDICTIONS, MODEL_INPUTS, BATCH_SIZE
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

def remove_equivalent_combinations(p1, p2):
    """
    Obsolete
    There are different stacks which have the same spectral behaviour.
    In these cases the NN can't "decide" which option to pick. This function
    rearanges p1 and p2 so that only on of the equivalent stacks is possible.

    Parameters
    ----------
    p1 : dict
    p2 : dict

    Returns
    -------
    (p1, p2) or
    (p2, p1)
    """
    if p1["particle_material"] == "Al" and p2["particle_material"] == "Au":
        return p2, p1
    else:
        return p1, p2

def pick_training_layers(crawler, param_dict):
    """
    This needs to be generalised for arbitrary keys at some point
    """
    #choose random parameters
    layer1 = {}
    layer2 = {}

    for key, val in MODEL_DISCRETE_PREDICTIONS.items():
        l1 = random.choice(val)
        l2 = random.choice(val)

        #arange the materials unambiguously
        if key == "particle_material":
            if l1 < l2:
                l1, l2 = l2, l1

        if key == "hole" and layer1["particle_material"] == layer2["particle_material"]:
            if l1 < l2:
                l1, l2 = l2, l1
                
        layer1[key] = l1
        layer2[key] = l2


    query1 = f"""SELECT simulations.m_file, simulations.adress
    FROM simulations
    INNER JOIN wire
    ON simulations.simulation_id = wire.simulation_id
    WHERE particle_material = '{layer1["particle_material"]}'
    AND wire.hole = '{layer1["hole"]}'
    AND meets_conditions = 1
    ORDER BY RANDOM()
    LIMIT 1"""
    #AND wire.width = wire.length

    query2 = f"""SELECT simulations.m_file, simulations.adress
    FROM simulations
    INNER JOIN wire
    ON simulations.simulation_id = wire.simulation_id
    WHERE particle_material = '{layer2["particle_material"]}'
    AND wire.hole = '{layer2["hole"]}'
    AND meets_conditions = 1
    ORDER BY RANDOM()
    LIMIT 1"""
    #AND wire.width = wire.length

    crawler.cursor.execute(query1)
    m_file, adress = crawler.cursor.fetchone()
    m1 = crawler.load_smat_npy(name=m_file, adress=adress)
    p1 = param_dict[m_file+adress+".npy"]

    crawler.cursor.execute(query2)
    m_file, adress = crawler.cursor.fetchone()
    m2 = crawler.load_smat_npy(name=m_file, adress=adress)
    p2 = param_dict[m_file+adress+".npy"]

    return m1 ,m2, p1, p2



def create_random_stack(crawler, param_dict):
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

    m1, m2, p1, p2 = pick_training_layers(crawler, param_dict)


    wav = np.linspace(
        WAVLENGTH_START,
        WAVLENGTH_STOP,
        NUMBER_OF_WAVLENGTHS)

    SiO2 = n_SiO2_formular(wav)

    l1 = MetaLayer(m1, SiO2, SiO2)
    l2 = MetaLayer(m2, SiO2, SiO2)

    phi = random.uniform(0,90)
    l1.rotate(phi)

    h = random.uniform(0.01, 0.3)
    spacer = NonMetaLayer(SiO2, height=h)

    s = Stack([l1, spacer, l2], wav, SiO2, SiO2)
    smat = s.build()

    spec_x = np.abs(smat[:, 0, 0])**2 / SiO2
    spec_y = np.abs(smat[:, 1, 1])**2 / SiO2

    p_stack = {
        'angle' : phi,
        'spacer_height': h,
        }

    return spec_x, spec_y, p1, p2, p_stack


def create_batch(size, mlb, crawler, param_dict):
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


    model_in = np.zeros((size, MODEL_INPUTS, 2))
    labels1 = []
    labels2 = []
    stack_params = []

    for i in range(size):

        #generate stacks until one doesn't block all incomming light
        while True:
            spec_x, spec_y, p1, p2, p_stack = create_random_stack(crawler, param_dict)

            if np.mean(spec_x) > 0.2 or np.mean(spec_y) > 0.2:
                break

        #save the input spectrum
        model_in[i] = np.stack((spec_x, spec_y), axis=1)

        #save the layer parameters which led to the spectrum
        label1 = [p1[key].strip() for key in MODEL_DISCRETE_PREDICTIONS]
        label2 = [p2[key].strip() for key in MODEL_DISCRETE_PREDICTIONS]

        labels1.append(label1)
        labels2.append(label2)

        stack_params.append((p1, p2, p_stack))

    #encode the labels
    enc1 = mlb.fit_transform(labels1)
    enc2 = mlb.fit_transform(labels2)

    model_out = np.concatenate((enc1, enc2), axis=1)

    return model_in, model_out, stack_params

def LabelBinarizer():
    discrete_params = ['Au', 'Al', 'holes', 'no holes']
    mlb = MultiLabelBinarizer(classes=np.array(discrete_params, dtype=object))
    mlb.fit_transform([['Au', 'holes']])
    return mlb

#%%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("src", metavar='src', type=str,
    	help="path to source directory containing .npy files")
    ap.add_argument("dst", metavar='dst', type=str,
        help="path to destination batch directory")
    ap.add_argument("-p", "--params", default="data/params.pickle",
    	help="path to the .pickle file containing the smat parameters")
    ap.add_argument("-n", "--number-of-batches", default=10, type=int)
    ap.add_argument("-db", "--database", default="data/NN_smats.db",
                        help="sqlite database containing the adresses")
    args = vars(ap.parse_args())


    print("[INFO] connecting to the db...")
    with sqlite3.connect(database=args['database']) as conn:
        crawler = Crawler(
            directory=args['src'],
            cursor=conn.cursor()
        )


    print("[INFO] loading data...")
    lb = LabelBinarizer()

    with open(args["params"], "rb") as f:
        param_dict = pickle.load(f)

    #make the dirctories for the samples
    if not os.path.exists(f"{args['dst']}/X"):
        os.mkdir(f"{args['dst']}/X")
        os.mkdir(f"{args['dst']}/Y")
        os.mkdir(f"{args['dst']}/params")


    for i in range(args["number_of_batches"]):
        print(f"[INFO] creating batch {i+1}/{args['number_of_batches']}")
        x, y, stack_params = create_batch(BATCH_SIZE, lb, crawler, param_dict)
        ts = str(datetime.now()).replace(" ", "_")
        np.save(f"{args['dst']}/X/{ts}.npy", x)
        np.save(f"{args['dst']}/Y/{ts}.npy", y)

        with open(f"{args['dst']}/params/{ts}.pickle", "wb") as f:
            pickle.dump(stack_params, f)

    print("[DONE]")
