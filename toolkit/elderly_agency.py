import shapefile
import numpy as np

def load(save=False):
    shape = shapefile.Reader("/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/MOHW_W84_ElderlyWelfareAgency_2017.shp")
    feature = shape.shapeRecords()
    record_list = []

    for rec in feature:
        if rec.record[4] == '臺南市':
            record_list.append(rec.record)

    record_list = np.array(record_list)
    if save:
        np.save("/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/Tainan_agency.npy", record_list)
        print("npy saved.")
    print("==========")
    print("{}/{} agencies found.".format(len(record_list), len(feature)))
    print("Type: {}".format(type(record_list)))
    print("==========")
