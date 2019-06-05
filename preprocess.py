import numpy as np
import jieba
import pickle
import datetime
from toolkit import tools

threshold = 300  # unit: meter
count_rec = 0
agency_path = "/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/Tainan_agency_with_wl_rain_station.npy"
flooding_rec_path = "/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/EMIC_record.npy"
wl_path = "/mnt/HDD_1/yanshuo/Waterlevel/water_level_2016_to_2019.pkl"
rain_path = "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl"

agency = tools.read_nparray(agency_path)
flooding_rec = tools.read_nparray(flooding_rec_path, allow_pickle=True)
with open(wl_path, 'rb') as f: water_level = pickle.load(f)
print("water level pkl loaded.")
with open(rain_path, 'rb') as f: rain_record = pickle.load(f)
print("rain station pkl loaded.")
flooding_near_agency_keys = []
flooding_near_agency_values = []

# Distance btw agency and EMIC within threshold
for record in flooding_rec:
    for ag in agency:
        distance = tools.cal_dist(record[6], record[7], float(ag[18]), float(ag[19]))
        if distance <= threshold:
            flooding_near_agency_keys.append(str(ag[2]))
            flooding_near_agency_values.append([ag[23], ag[24], record[0], record[6], record[7], record[11]])
            count_rec = count_rec + 1

print("{}/{} flooding record within {} meter.".format(count_rec, len(flooding_rec), threshold))

# Determine if the description contains words like "cm" or "m"
# description: array index [11]
flooding_near_agency_with_label = []
for record in flooding_near_agency_values:
    words = jieba.cut(str(record[-1]))
    words = list(words)
    
    if "公分" in words:
        flooding_near_agency_with_label.append(record)
    if "公尺" in words:
        flooding_near_agency_with_label.append(record)

print("{}/{} with 'cm' or 'm', but requiring human check.".format(len(flooding_near_agency_with_label), len(flooding_near_agency_keys)))

### wl and rain station pair ###
def pair_time_height(start_time, end_time, recordings, flag=None):
    record_in_range = {}
    if flag == 'wl':
        for time in recordings:
            if end_time > time and time > start_time:
                record_in_range[time] = recordings[time][0]
    if flag == 'rain':
        for time in recordings:
            if end_time > time and time > start_time:
                record_in_range[time] = recordings[time]
    
    times = []
    height = []
    for time in sorted(record_in_range.keys()):
        times.append(time)
        height.append(record_in_range[time])
    
    return times, height

unavailable_wl = 0
unavailable_rain = 0
total_unavailable = 0
for keys, values in zip(flooding_near_agency_keys, flooding_near_agency_values):
    
    # WL
    water_level_station_records = water_level[values[0]]
    end_time = values[2]
    start_time = values[2] - datetime.timedelta(hours=72)
    times_wl, height_wl = pair_time_height(start_time, end_time, water_level_station_records, flag='wl')
    if not times_wl: unavailable_wl += 1; total_unavailable += 1; continue
    
    # Rain
    rain_station_records = rain_record[values[1]]
    times_rain, height_rain = pair_time_height(start_time, end_time, rain_station_records, flag='rain')
    if not times_rain: unavailable_rain += 1; total_unavailable += 1; continue
    
    tools.plot_trend_line(times_wl, height_wl, times_rain, height_rain, keys)

print("unavailable water level: {} || unavailable rain station: {} || unavailable data: {}/{}".format(unavailable_wl, unavailable_rain, total_unavailable, len(flooding_near_agency_keys)))
