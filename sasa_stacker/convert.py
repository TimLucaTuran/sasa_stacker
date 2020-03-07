from sasa_db.crawler import Crawler
import sqlite3
import pickle
import numpy as np
import scipy
import argparse
#%%
def convert_to_npy(crawler, ids, dst):
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
        with open(f"{dst}/params.pickle", "rb") as f:
            param_dict = pickle.load(f)
    except FileNotFoundError:
        with open(f"{dst}/params.pickle", "w+") as f:
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
        np.save("{}/{}".format(dst, fullname), smat)
        #write params to dict
        params = crawler.extract_params(id)
        param_dict[fullname] = params

    #pickle param_dict
    with open(f"{dst}/params.pickle", "wb") as f:
        pickle.dump(param_dict, f)
#%%
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src', metavar='src', type=str,
                        help='src dir of the .mat files')
    ap.add_argument('dst', metavar='dst', type=str,
                        help='dst dir for the .npy files')
    ap.add_argument("-db", "--database",
                        help="sqlite database containing the adresses")
    args = vars(ap.parse_args())
    print(args)

    conn = sqlite3.connect(args['database'])
    cursor = conn.cursor()
    crawler = Crawler(directory=args['src'], cursor=cursor)


    cursor.execute("""SELECT simulation_id FROM wire""")
    ids = [id[0] for id in cursor.fetchall()]
    print(max(ids) - min(ids))
    #%%
    convert_to_npy(crawler, ids, args['dst'])
    #%%
    #mat = crawler.find_smat("NN_wires_150nm_anti")
    #mat.shape
    #mat = np.swapaxes(mat, 2,3)
    #scipy.io.savemat("../meta_material_databank/collected_mats/NN_wires_150nm_anti_Daten_gesamt.mat", {'SMAT_' : mat})
