import pandas as pd
import numpy as np
import datetime
import pickle
from multiprocessing import Process, Manager

filenames = ['water_level_2016.csv', 'water_level_2017.csv', 'water_level_2018.csv', 'water_level_201901.csv', 'water_level_201902.csv', 'water_level_201903.csv', 'water_level_201904.csv']

def iso_format_to_time(dt_str):
    try:
        time = datetime.datetime.strptime(dt_str.split("+")[0].split(".")[0], "%Y-%m-%dT%H:%M:%S")
    except:
        time = datetime.datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
    return time

def save_pickle(x, filename):
    with open(filename, 'wb') as f: pickle.dump(x, f)

def Func(filename, wl_dict):
    print(filename)
    global removal_count
    file_csv_path = pd.read_csv("/mnt/HDD_1/yanshuo/Waterlevel/{}".format(filename))
    file_csv = np.array(file_csv_path)

    for record in file_csv:
        if record[0] not in wl_dict: wl_dict[record[0]] = {}
        if record[3] < 0: removal_count += 1; continue
        wl_dict[str(record[0])][iso_format_to_time(record[2])] = [record[3], record[1]]

removal_count = 0
jobs = []
def load(save=False):
    
    print("Reading file...")
    
    manager = Manager()
    wl_dict = manager.dict()

    for filename in filenames:
        
        p = Process(target=Func, args=(filename, wl_dict))
        jobs.append(p)
        p.start()
        
    for pp in jobs:
        pp.join()

    print("Remove {} record.".format(removal_count))
    if save:
        save_pickle(wl_dict, "/mnt/HDD_1/yanshuo/Waterlevel/water_level_2016_to_2019.pkl")
        print("water_level_2016_to_2019.pkl saved.")
