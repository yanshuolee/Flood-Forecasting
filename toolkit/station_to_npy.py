import pandas as pd
import numpy as np
import coordinate

rain_station_csv = pd.read_csv("/mnt/HDD_1/yanshuo/Rainfall/rain_station.csv")
rain_station_csv = np.array(rain_station_csv)
rain_station = []

water_station_csv = pd.read_csv("/mnt/HDD_1/yanshuo/Waterlevel/water_level_station.csv")
water_station_csv = np.array(water_station_csv)
water_station = []

for d in rain_station_csv:
    if d[4] == '臺南市':
        xy = coordinate.Cal_lonlat_To_twd97(d[2], d[3])
        tmp = d
        tmp[2] = xy[0]
        tmp[3] = xy[1]
        rain_station.append(tmp)

rain_station = np.array(rain_station)
print("{} out of {} rain station in Tainan.".format(len(rain_station), len(rain_station_csv)))
# np.save("/mnt/HDD_1/yanshuo/Rainfall/rain_station.npy", rain_station)

for d in water_station_csv:
        if d[8] == '臺南市':
                xy = coordinate.Cal_lonlat_To_twd97(d[5], d[6])
                tmp = d
                tmp[5] = xy[0]
                tmp[6] = xy[1]
                water_station.append(tmp)

water_station = np.array(water_station)
print("{} out of {} water level station in Tainan.".format(len(water_station), len(water_station_csv)))
# np.save("/mnt/HDD_1/yanshuo/Waterlevel/water_level_station.npy", water_station)
