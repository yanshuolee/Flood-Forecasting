import numpy as np
import jieba
import pickle
import datetime
from toolkit import tools
from shapely.geometry import Point
from topography import crop

window_size = 6
stride = 1
n_class = [0,0,0,0,0,0]
data = []
topo_data = []
label = []

rain_path = "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl"
with open(rain_path, 'rb') as f: rain_record = pickle.load(f)
print("rainfall pkl loaded.")
map_path = "/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/maps.pkl"
with open(map_path, 'rb') as f: maps = pickle.load(f)
print("map pkl loaded.")

tainan_agency_path = "/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/Tainan_agency.npy"

tainan_agency = tools.read_nparray(tainan_agency_path)
tainan_agency_with_rain_index = tools.pair_rain(tainan_agency)
print("{} agancy in Tainan.".format(len(tainan_agency_with_rain_index)))

start_date_1 = datetime.datetime(2016, 1, 1, 0, 0)
end_date_1 = datetime.datetime(2018, 1, 1, 0, 0)
start_date = datetime.datetime(2018, 11, 1, 0, 0)
end_date = datetime.datetime(2019, 5, 1, 0, 0)

def is_valid(rec, timestamp):
    validity = True
    cumulative_rain = sum(rec)
    if cumulative_rain >= 350:
        label_name = '350'
    elif cumulative_rain >= 250:
        label_name = '250'
    elif cumulative_rain >= 150:
        label_name = '150'
    else:
        label_name = None
        validity = False
    
    return label_name, validity

def find_label(label_name, point):
    index = str(window_size)+"h"+label_name+".json"
    for level in maps[index]:
        for polygon in maps[index][level]:
            if polygon.contains(point):
                return level
    
    raise Exception("Not found!")

def to_category(str_label):
    if str_label == "0-0.3":
        n_class[0] += 1
        return 0
    elif str_label == "0.3-0.5":
        n_class[1] += 1
        return 1
    elif str_label == "0.5-1":
        n_class[2] += 1
        return 2
    elif str_label == "1-2":
        n_class[3] += 1
        return 3
    elif str_label == "2-3":
        n_class[4] += 1
        return 4
    else:
        n_class[5] += 1
        return 5

tmp_time = []
tmp_height = []
def slide_recording(times, heights, point, X, Y):
    for i in range(0, len(times)-window_size+1, stride):
        timestamp = times[i:i+window_size] # length of window size
        rec = heights[i:i+window_size] # length of window size
        label_name, validity = is_valid(rec, timestamp) # invalid condition: < 150 mm
        
        if validity:
            str_label = find_label(label_name, point)
            data.append(rec)
            int_label = to_category(str_label)
            label.append(int_label)
            topo_data.append(crop.foo(X, Y))
        else:
            tmp_time.append(timestamp)
            tmp_height.append(rec)

for rec in tainan_agency_with_rain_index:
    rain_id = rec[23]
    agency_X = float(rec[18])
    agency_Y = float(rec[19])
    point = Point(agency_X, agency_Y)
    record_in_Dec = {}
    record_in_range = {}
    for time in rain_record[rain_id]:
        if end_date_1 > time and time > start_date_1:
            record_in_Dec[time] = rain_record[rain_id][time]
        if end_date > time and time > start_date:
            record_in_range[time] = rain_record[rain_id][time]
    
    times = []
    height = []
    for time in sorted(record_in_range.keys()):
        times.append(time)
        height.append(record_in_range[time])

    times_Dec = []
    height_Dec = []
    for time in sorted(record_in_Dec.keys()):
        times_Dec.append(time)
        height_Dec.append(record_in_Dec[time])
    
    slide_recording(times, height, point, agency_X, agency_Y)
    slide_recording(times_Dec, height_Dec, point, agency_X, agency_Y)

print("data: {}".format(len(data)))
print("label: {}".format(len(label)))
print("topo data: {}".format(len(topo_data)))
print("=====")
print("0-0.3: {} 0.3-0.5: {} 0.5-1: {} 1-2: {} 2-3: {} >3: {}".format(n_class[0], n_class[1], n_class[2], n_class[3], n_class[4], n_class[5]))
print("Total: {}".format(n_class[0]+n_class[1]+n_class[2]+n_class[3]+n_class[4]+n_class[5]))

print("")

data = np.array(data)
topo_data = np.array(topo_data)
label = np.array(label)
np.save("/mnt/HDD_1/yanshuo/Task_1/data.npy", data)
np.save("/mnt/HDD_1/yanshuo/Task_1/data_topo.npy", topo_data)
np.save("/mnt/HDD_1/yanshuo/Task_1/label.npy", label)