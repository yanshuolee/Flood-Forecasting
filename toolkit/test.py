# import pickle
# import numpy

# wl_path = "/mnt/HDD_1/yanshuo/Waterlevel/water_level_2016_to_2019.pkl"
# rain_path = "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl"

# with open(wl_path, 'rb') as f: water_level = pickle.load(f)
# with open(rain_path, 'rb') as f: rain_record = pickle.load(f)



# import shapefile

# shape = shapefile.Reader("/mnt/HDD_1/yanshuo/潛視圖/臺南市/SHP/6h150.shp")
# feature = shape.shapeRecords()[0]



# import pandas as pd
# import os

# # df = pd.read_json("/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/24h650.json")
# df_prev = pd.read_json("/home/yanshuo/Desktop/Flood/臺南淹水潛勢/geojson/台南市350.json")

# directories = os.listdir("/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/")
# for d in directories:
    
#     df = pd.read_json("/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/{}".format(d))
#     print(d)
#     for i in range(len(df['features'])):
#         print(df['features'][i]['properties'])


# import numpy as np
# save_path = "/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/"
# filename = "elder"
# labeled = np.load(save_path+filename+"_labeled_value.npy", allow_pickle=True)
# unlabeled = np.load(save_path+filename+"_unlabeled_key.npy")

