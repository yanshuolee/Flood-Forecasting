import numpy as np
import jieba
import pickle
import datetime
from toolkit import tools
from shapely.geometry import Point

agency_path = "/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/Tainan_agency_with_wl_rain_station.npy"
flooding_rec_path = "/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/EMIC_record.npy"
wl_path = "/mnt/HDD_1/yanshuo/Waterlevel/water_level_2016_to_2019.pkl"
rain_path = "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl"
map_path = "/mnt/HDD_1/yanshuo/潛視圖/臺南市/JSON/maps.pkl"

agency = tools.read_nparray(agency_path)
flooding_rec = tools.read_nparray(flooding_rec_path, allow_pickle=True)
with open(map_path, 'rb') as f: maps = pickle.load(f)
print("map pkl loaded.")

n_class = [0,0,0,0,0,0]
time_before=72
window_size=6
stride=1
data = []
label = []

def pair_time_height(recording, rain_recordings, flag=None):
    end_time = recording[0]
    start_time = end_time - datetime.timedelta(hours=time_before)
    
    record_in_range = {}
    
    if flag == 'rain':
        for time in rain_recordings:
            if end_time > time and time > start_time:
                record_in_range[time] = rain_recordings[time]
    
    times = []
    heights = []
    for time in sorted(record_in_range.keys()):
        times.append(time)
        heights.append(record_in_range[time])

    return times, heights

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
    
    # time_delta = timestamp[-1] - timestamp[0]
    # if time_delta.seconds != (window_size-1)*60*60:
    #     check_data.append([rec])
    #     check_label.append([timestamp])
    
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

def slide_recording(times, heights, point):
    for i in range(0, time_before-window_size+1, stride):
        timestamp = times[i:i+window_size] # length of window size
        rec = heights[i:i+window_size] # length of window size
        label_name, validity = is_valid(rec, timestamp) # invalid condition: < 150 mm
        
        if validity:
            str_label = find_label(label_name, point)
            data.append(rec)
            int_label = to_category(str_label)
            label.append(int_label)

def main():
    
    '''with open(wl_path, 'rb') as f: water_level = pickle.load(f)
    print("water level pkl loaded.")'''
    with open(rain_path, 'rb') as f: rain_record = pickle.load(f)
    print("rain station pkl loaded.")

    # append nearest rain station with EMIC record.
    flood_rec_with_rain = tools.pair_rain_preprocess(flooding_rec)
    
    unavailable_rec = 0
    unavailable_rec_times = []
    unavailable_rec_heights = []
    for recording in flood_rec_with_rain:
        
        point = Point(recording[6], recording[7]) # get flooding location position
        rain_recordings = rain_record[recording[20]] # get corresponding rain recording
        times, heights = pair_time_height(recording, rain_recordings, flag='rain') # collect rainfall data btw start and end
        
        if len(times) != 72: # Error handling
            unavailable_rec += 1
            unavailable_rec_times.append(times)
            unavailable_rec_heights.append(heights)
            continue
        
        slide_recording(times, heights, point)
    
    print("unavailable: {} total: {}".format(unavailable_rec, len(flood_rec_with_rain)))
    print("=====")
    print("0-0.3: {} 0.3-0.5: {} 0.5-1: {} 1-2: {} 2-3: {} >3: {}".format(n_class[0], n_class[1], n_class[2], n_class[3], n_class[4], n_class[5]))
    print("Total: {}".format(n_class[0]+n_class[1]+n_class[2]+n_class[3]+n_class[4]+n_class[5]))

    count = {}
    tmp = [len(t) for t in unavailable_rec_heights]
    for t in tmp:
        if t not in count:
            count[t] = 0
        else:
            count[t] = count[t] + 1

    return data, label


if __name__ == "__main__":
    data, label = main()