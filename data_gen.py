import pickle
import sys
import sqlite3
sys.path.insert(0, "/home/tim/Desktop/Uni/HIWI/meta_material_databank")
from crawler import Crawler

#%%
if __name__ == '__main__':
    print("[INFO] connecting to the databank")

    conn = sqlite3.connect('/home/tim/Desktop/Uni/HIWI/meta_material_databank/meta_materials.db')
    cursor = conn.cursor()
    crawler = Crawler(directory='/home/tim/Desktop/Uni/HIWI/collected_mats',
                cursor = cursor)

    cursor.execute("""SELECT simulation_id FROM simulations
                   WHERE angle_of_incidence=0
                   AND geometry = 'square'
                   AND wavelength_start = 0.5
                   AND wavelength_stop = 1
                   AND spectral_points =  128""")
    ids = [id[0] for id in cursor.fetchall()]

    print("[INFO] saving smats and parameters")
    crawler.convert_to_npy(ids)
    
