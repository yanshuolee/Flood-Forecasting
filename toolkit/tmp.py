import pickle
import numpy

wl_path = "/mnt/HDD_1/yanshuo/Waterlevel/water_level_2016_to_2019.pkl"
rain_path = "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl"

with open(wl_path, 'rb') as f: water_level = pickle.load(f)
with open(rain_path, 'rb') as f: rain_record = pickle.load(f)


print("t")