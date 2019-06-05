import numpy as np
import math
import pylab as plt

def read_nparray(filePath, **args):
    data = np.load(filePath, **args)
    return data

def cal_dist(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

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

def plot_trend_line(times_wl, height_wl, times_rain, height_rain, agency_name):
    plt.figure(figsize=(10,5))
    plot_wl, = plt.plot(times_wl, height_wl)
    plot_rain, = plt.plot(times_rain, height_rain)
    plt.title(agency_name, fontproperties="DFKai-SB")
    plt.legend(handles = [plot_wl, plot_rain], labels = ["water level station", "rain station"], loc = 'best' )
    plt.savefig("{}/{}.png".format("/mnt/HDD_1/yanshuo/trend_line", agency_name))
    plt.clf()
    print("{} plotted.".format(agency_name))

if __name__ == "__main__":
    # a=read_nparray("/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/EMIC_record.npy", allow_pickle=True)
    # b=read_nparray("/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/Tainan_agency.npy")
    # append_wl_rain_station_to_agency(b)
    print("d")