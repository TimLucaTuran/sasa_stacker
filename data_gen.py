import sys
sys.path.insert(0, "../meta_material_databank")

import pickle
import sqlite3
from crawler import Crawler

#%%
if __name__ == '__main__':
    print("[INFO] connecting to the databank")

    conn = sqlite3.connect("../meta_material_databank/NN_smats.db")
    cursor = conn.cursor()
    crawler = Crawler(directory="../meta_material_databank/collected_mats",
                cursor = cursor)

    cursor.execute("""SELECT simulation_id FROM simulations
                   WHERE angle_of_incidence=0
                   AND geometry = 'square'
                   AND wavelength_start = 0.5
                   AND wavelength_stop = 1
                   AND spectral_points =  128""")
    ids = [id[0] for id in cursor.fetchall()]

    print("[INFO] converting {} IDs to .npy/.pickle".format(len(ids)))
    crawler.convert_to_npy(ids)
