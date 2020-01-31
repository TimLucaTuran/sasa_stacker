import sys
sys.path.insert(0, "../meta_material_databank")

from crawler import Crawler
import sqlite3
import pickle
import numpy as np
import scipy
#%%
def convert_to_npy(crawler, ids):
    """
    Loads the .mat files for all the IDs, splits them into one file per ID
    and saves them as .npy for quicker access
    Also extracts the parameters of every ID and saves them to a .pickle file

    Parameters
    ----------
    ids : list
    """
    #load param_dict
    try:
        with open("data/params.pickle", "rb") as f:
            param_dict = pickle.load(f)
    except FileNotFoundError:
        with open("data/params.pickle", "w+") as f:
            param_dict = {}

    for id in ids:
        print("converting id: ", id)
        #save smat
        query = 'SELECT m_file, adress FROM simulations WHERE simulation_id = {}'.format(id)
        crawler.cursor.execute(query)
        row = crawler.cursor.fetchone()
        name = row[0]
        adress = row[1]
        if type(adress) is str:
            adress = eval(adress,{"__builtins__":None})

        fullname = "{}{}.npy".format(name, adress)
        smat = crawler.find_smat(name, adress)
        np.save("data/smat_data/{}".format(fullname), smat)
        #write params to dict
        params = crawler.extract_params(id)
        param_dict[fullname] = params

    #pickle param_dict
    with open("data/params.pickle", "wb") as f:
        pickle.dump(param_dict, f)
#%%
conn = sqlite3.connect('../meta_material_databank/NN_smats.db')
cursor = conn.cursor()
crawler = Crawler(directory='../meta_material_databank/collected_mats', cursor=cursor)


cursor.execute("""SELECT simulation_id FROM wire
               WHERE width=150 """)
ids = [id[0] for id in cursor.fetchall()]
print(max(ids) - min(ids))
#%%
convert_to_npy(crawler, ids)
#%%
#mat = crawler.find_smat("NN_wires_150nm_anti")
#mat.shape
#mat = np.swapaxes(mat, 2,3)
#scipy.io.savemat("../meta_material_databank/collected_mats/NN_wires_150nm_anti_Daten_gesamt.mat", {'SMAT_' : mat})
