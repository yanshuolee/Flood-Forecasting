# import pickle
# import numpy

# wl_path = "/mnt/HDD_1/yanshuo/Waterlevel/water_level_2016_to_2019.pkl"
# rain_path = "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl"

# with open(wl_path, 'rb') as f: water_level = pickle.load(f)
# with open(rain_path, 'rb') as f: rain_record = pickle.load(f)




# import shapefile

# shape = shapefile.Reader("/mnt/HDD_1/yanshuo/潛視圖/臺南市/SHP/6h150.shp")
# feature = shape.shapeRecords()[0]



import pandas as pd
df = pd.read_json("/home/yanshuo/Downloads/6h150.json")
print("t")