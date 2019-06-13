import os
import pandas as pd
from shapely.geometry.polygon import Polygon
import pickle

def save_pickle(x, filename):
    with open(filename, 'wb') as f: pickle.dump(x, f)

def load(save=False):
    """
    The "maps" dict structure defined as follows

    Filename 1
        - level 1
            - coordinate 1
            - coordinate 2
            - ...
            - coordinate n
        - level 2
            ...
        - level 6
    Filename 2
        - level 1
        ...

    [Type]
    Filename: string
    level 1: string
    coordinate: Polygon object
    """
    directories = [d for d in os.listdir("/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/") if d.endswith('.json')]
    maps = {}

    for d_name in directories:
        
        df = pd.read_json("/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/{}".format(d_name))
        print(d_name)
        maps[d_name] = {}

        for i in range(len(df['features'])):
            coordinates = df['features'][i]['geometry']['coordinates']
            try:
                level = df['features'][i]['properties']['type']
            except:
                level = df['features'][i]['properties']['Type']
            list_of_polygon_coordinates = [Polygon(coord[0]) for coord in coordinates]
            maps[d_name][level] = list_of_polygon_coordinates
    
    if save:
        save_pickle(maps, "/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/maps.pkl")
        print("maps.pkl saved.")
    
    return maps
