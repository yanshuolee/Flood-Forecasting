import pickle
import numpy as np
import pandas as pd
import datetime
import pickle
import os

def save_pickle(x, filename):
    with open(filename, 'wb') as f: pickle.dump(x, f)

def load(save=False):
    
    rain_dict = {}
    ### processing 1998-2017.txt ###
    print("processing 1998-2017.txt")

    def to_time(txt):

        try:
            time = datetime.datetime.strptime(txt, "%Y%m%d%H")
        except:
            txt = txt[:-2]+'0'
            time = datetime.datetime.strptime(txt, "%Y%m%d%H")
            time += datetime.timedelta(days=1)
        
        return time

    txt_file_path = "/mnt/HDD_1/yanshuo/Rainfall/Rain_1998-2017.txt"
    special_value = 0
    with open(txt_file_path, "r") as reader:
        r = reader.readline()
        for r in reader.readlines():
            record = r.split(' ')
            if record[0] not in rain_dict: rain_dict[record[0]] = {}
            if float(record[-1]) < 0: special_value += 1; continue
            rain_dict[record[0]] [to_time(record[1])] = float(record[-1])


    ### type 1: processing 2018-11~.csv(full info) ###
    print("processing 2018-11~.csv")

    def to_time_type_1(txt):
        try:
            time = datetime.datetime.strptime(txt, "%d/%m/%Y %H:%M:%S")
        except:
            time = datetime.datetime.strptime(txt, "%Y-%m-%d %H:%M:%S")
        return time

    filenames = ['rain_201811.csv', 'rain_201812.csv', 'rain_201901.csv', 'rain_201902.csv', 'rain_201903.csv', 'rain_201904.csv']
    special_value_type_1 = 0
    for filename in filenames:
        
        print(filename)
        file_csv_path = pd.read_csv("/mnt/HDD_1/yanshuo/Rainfall/{}".format(filename))
        file_csv = np.array(file_csv_path)

        for record in file_csv:
            timestamp = to_time_type_1(record[1])
            if timestamp.time().minute == 0:
                if record[0] not in rain_dict: rain_dict[record[0]] = {}
                level = 0.0 if record[4] == -998.0 else record[4]
                if record[4] < 0 and record[4]!=-998.0: special_value_type_1 += 1; continue
                rain_dict[record[0]][timestamp] = level


    ### type 2: processing 2018-1~2018-10.csv(metro and auto) ###
    print("processing 2018-1~2018-10.csv")

    def to_time_type_2(txt):
        time = datetime.datetime.strptime(txt, "%Y-%m-%d %H:%M:%S")
        return time

    filenames_2018 = os.listdir("/mnt/HDD_1/yanshuo/Rainfall/Rain_2018")
    special_value_type_2 = 0
    for filename in filenames_2018:
        
        print(filename)
        file_csv_path = pd.read_csv("/mnt/HDD_1/yanshuo/Rainfall/Rain_2018/{}".format(filename), encoding='big5')
        file_csv = np.array(file_csv_path)

        for record in file_csv:
            if record[1] not in rain_dict: rain_dict[record[1]] = {}
            if record[3]<0: special_value_type_2 += 1; continue
            rain_dict[record[1]][to_time_type_2(record[0])] = record[3]

    print("Removed data")
    print("1998-2017.txt: {} || 2018-11~.csv: {} || 2018-1~2018-10.csv: {}".format(special_value, special_value_type_1, special_value_type_2))

    if save:
        save_pickle(rain_dict, "/mnt/HDD_1/yanshuo/Rainfall/Rainfall_ALL.pkl")
        print("Rainfall_ALL.pkl saved.")
