import numpy as np
import math
import pylab as plt
import jieba
import matplotlib.font_manager as mfm
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

def read_nparray(filePath, **args):
    data = np.load(filePath, **args)
    return data

def cal_dist(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def get_rain_station(X, Y):
    rain_path = "/mnt/HDD_1/yanshuo/Rainfall/rain_station.npy"
    rain_station = read_nparray(rain_path, allow_pickle=True)
    rain_station = rain_station.tolist()

    min_dist = 9999999
    for i in range(len(rain_station)):
        dist = cal_dist(X, Y, rain_station[i][2], rain_station[i][3])
        if dist < min_dist:
            min_dist = dist
            index = i
    return rain_station[index][0]

def pair_wl(agency_list, verbose=0):
    wl_path = "/mnt/HDD_1/yanshuo/Waterlevel/water_level_station.npy"
    water_level = read_nparray(wl_path, allow_pickle=True)
    water_level = water_level.tolist()
    agency = agency_list
    for j in range(len(agency)):
        min_dist = 9999999
        for i in range(len(water_level)):
            dist = cal_dist(float(agency[j][18]), float(agency[j][19]), water_level[i][5], water_level[i][6])
            if dist < min_dist:
                min_dist = dist
                index = i
        agency_list[j].append(water_level[index][0])
        if verbose:
            print("Distance from {} to {} is {}.".format(agency_list[j][2], water_level[index][4], min_dist))
    
    return agency_list
        
def pair_rain(agency_list, verbose=0):
    rain_path = "/mnt/HDD_1/yanshuo/Rainfall/rain_station.npy"
    rain_station = read_nparray(rain_path, allow_pickle=True)
    rain_station = rain_station.tolist()
    agency_list = agency_list.tolist()
    agency = agency_list
    for j in range(len(agency)):
        min_dist = 9999999
        for i in range(len(rain_station)):
            dist = cal_dist(float(agency[j][18]), float(agency[j][19]), rain_station[i][2], rain_station[i][3])
            if dist < min_dist:
                min_dist = dist
                index = i
        agency_list[j].append(rain_station[index][0])
        if verbose:
            print("Distance from {} to {} is {}.".format(agency_list[j][2], rain_station[index][1], min_dist))

    return agency_list

def append_wl_rain_station_to_agency(agency_list):
    agency_list = agency_list.tolist()
    agency_list = pair_wl(agency_list)
    agency_list = pair_rain(agency_list)
    agency_list = np.array(agency_list)
    np.save("/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/Tainan_agency_with_wl_rain_station.npy", agency_list)

def pair_rain_preprocess(recordings, verbose=0):
    rain_path = "/mnt/HDD_1/yanshuo/Rainfall/rain_station.npy"
    rain_station = read_nparray(rain_path, allow_pickle=True)
    rain_station = rain_station.tolist()
    recordings = recordings.tolist()
    rec = recordings
    for j in range(len(rec)):
        min_dist = 9999999
        for i in range(len(rain_station)):
            dist = cal_dist(float(rec[j][6]), float(rec[j][7]), rain_station[i][2], rain_station[i][3])
            if dist < min_dist:
                min_dist = dist
                index = i
        recordings[j].append(rain_station[index][0])
        if verbose:
            print("Distance from {} to {} is {}.".format(recordings[j][2], rain_station[index][1], min_dist))

    recordings = np.array(recordings)
    return recordings

def plot_trend_line(times_wl, height_wl, times_rain, height_rain, agency_name, index):
    font_path = "/mnt/HDD_1/yanshuo/Flooding/toolkit/kaiu.ttf"
    plt.figure(figsize=(10,5))
    plot_wl, = plt.plot(times_wl, height_wl)
    plot_rain, = plt.plot(times_rain, height_rain)
    prop = mfm.FontProperties(fname=font_path)
    plt.title(agency_name, fontproperties=prop, fontsize='large')
    plt.legend(handles = [plot_wl, plot_rain], labels = ["water level station", "rain station"], loc = 'best' )
    plt.savefig("{}/{}_{}.png".format("/mnt/HDD_1/yanshuo/trend_line", index, agency_name))
    plt.clf()
    plt.close()
    print("{}_{} plotted.".format(index, agency_name))

def human_labeling(keys, values, unlabeled_keys, unlabeled_values):
    print("=== Starting labeling. Type 0 if it is unlabeled, else 1. ===")
    
    remove_index = []
    for i, j in enumerate(zip(keys, values)):
        print("{}/{}:".format(i+1, len(keys)))
        print("{}".format(j[1][5]))
        print("")
        
        while True:
            inputs = input("(0: unlabeled 1: labeled) => ")
            if inputs == "": continue
            inputs = int(inputs)
            if inputs == 0 or inputs == 1: break
        
        if not inputs:
            unlabeled_keys.append(j[0])
            unlabeled_values.append(j[1])
            remove_index.append(i)
    
    keys = np.array(keys)
    values = np.array(values)
    keys = np.delete(keys, remove_index)
    values = np.delete(values, remove_index,axis=0)

    unlabeled_keys = np.array(unlabeled_keys)
    unlabeled_values = np.array(unlabeled_values)

    return keys, values, unlabeled_keys, unlabeled_values

def labelling(keys, values, filename):
    # Determine if the description contains words like "cm" or "m"
    # description: array index [11]

    labeled_keys = []
    labeled_values = []
    unlabeled_keys = []
    unlabeled_values = []

    for key, record in zip(keys, values):
        words = jieba.cut(str(record[-1]))
        words = list(words)
        
        if "公分" in words:
            labeled_keys.append(key)
            labeled_values.append(record)
        elif "公尺" in words:
            labeled_keys.append(key)
            labeled_values.append(record)
        else:
            unlabeled_keys.append(key)
            unlabeled_values.append(record)
    
    print("=== Human checking ===")
    _labeled_keys, _labeled_values,_unlabeled_keys, _unlabeled_values = human_labeling(labeled_keys, labeled_values, unlabeled_keys, unlabeled_values)
    print("labeled: {}, unlabeled: {}, {} in total.".format(len(_labeled_values), len(_unlabeled_values), len(keys)))

    while True:
        inputs = input("Save? Yes(1)/No(0): ")
        inputs = int(inputs)
        if inputs == 0 or inputs == 1: break
    
    if inputs == 1:
        save_path = "/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/"
        np.save(save_path+filename+"_labeled_key.npy", _labeled_keys)
        np.save(save_path+filename+"_labeled_value.npy", _labeled_values)
        np.save(save_path+filename+"_unlabeled_key.npy", _unlabeled_keys)
        np.save(save_path+filename+"_unlabeled_value.npy", _unlabeled_values)
        print("Saved!")

def to_label(flood_height, units):
    if units == "公分":
        if 0 <= flood_height < 30:
            return 0
        elif 30 <= flood_height < 50:
            return 1
        elif 50 <= flood_height < 100:
            return 2
        elif 100 <= flood_height < 200:
            return 3
        elif 200 <= flood_height < 300:
            return 4
        elif 300 <= flood_height:
            return 5
    elif units == "公尺":
        if 0 <= flood_height < 0.3:
            return 0
        elif 0.3 <= flood_height < 0.5:
            return 1
        elif 0.5 <= flood_height < 1:
            return 2
        elif 1 <= flood_height < 2:
            return 3
        elif 2 <= flood_height < 3:
            return 4
        elif 3 <= flood_height:
            return 5

def human_labelling(sentence):
    print("Sentence: {}".format(sentence))
    while True:
        inputs = input("Class => ")
        if inputs == "": continue
        inputs = int(inputs)
        if 0 <= inputs <= 5: break
    
    return inputs

def sentence_to_label(words):
    sentence = words
    words = jieba.cut(words)
    words = list(words)
    
    start_list = ["淹水", "積水", "進水", "水淹", "淹", "積淹水", "進水", "積水約"]
    end_list = ["公分", "公尺"]
    
    start_index = None
    end_index = None
    
    for start_word in start_list:
        if start_word in words:
            start_index = words.index(start_word)
            break
            
    for end_word in end_list:
        if end_word in words:
            end_index = words.index(end_word)
            break
            
    if start_index is not None and end_index is not None:
        for i in words[start_index:end_index+1]:
            try:
                height = int(i)
                class_ = to_label(height, words[end_index])
                print("Sentence: {}".format(sentence))
                print("Height={} ==> Class={}".format(height, class_))
                return class_
            except:
                pass
        return human_labelling(sentence)
    else:
        return human_labelling(sentence)

def get_callbacks(modelPath, logPath):
	csv_logger = CSVLogger(filename=logPath)
	model_checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')
	model_earlystop = EarlyStopping(monitor="val_loss", patience=20)
	callbacks = [model_earlystop, csv_logger, model_checkpoint]
	return callbacks

