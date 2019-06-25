import numpy as np
import pickle
import datetime
from toolkit import tools
from topography import crop

data_path = "/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/"
keys = tools.read_nparray(data_path+"ALL"+"_labeled_key.npy", allow_pickle=True)
values = tools.read_nparray(data_path+"ALL"+"_labeled_value.npy", allow_pickle=True)
print("No. of data: {}".format(len(keys)))

rain_path = "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl"
with open(rain_path, 'rb') as f: rain_record = pickle.load(f)
print("rainfall pkl loaded.")

before_hour = 6
data = []
topo_data = []
label = []

tmp_data = []
count = 0
for rec in values:
    # data
    record_in_range = {}
    end_time = rec[2]
    start_time = end_time - datetime.timedelta(hours=before_hour)
    rain_station_name = tools.get_rain_station(rec[3], rec[4])
    for time in rain_record[rain_station_name]:
        if end_time > time and time > start_time:
            record_in_range[time] = rain_record[rain_station_name][time]
    times = []
    heights = []
    for time in sorted(record_in_range.keys()):
        times.append(time)
        heights.append(record_in_range[time])
    
    if len(heights) != before_hour:
        count += 1
        tmp_data.append(heights)
        continue
    
    data.append(heights)
    topo_data.append(crop.foo(rec[3], rec[4]))
    # label
    class_ = tools.sentence_to_label(rec[5])
    label.append(class_)
    count += 1

print("")

data = np.array(data)
topo_data = np.array(topo_data)
label = np.array(label)
np.save("/mnt/HDD_1/yanshuo/Task_2/data.npy", data)
np.save("/mnt/HDD_1/yanshuo/Task_2/data_topo.npy", topo_data)
np.save("/mnt/HDD_1/yanshuo/Task_2/label.npy", label)