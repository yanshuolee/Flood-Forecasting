import pandas as pd
import numpy as np

def load(save=False):
    xl = pd.read_excel("/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/EMIC2016-2019台南市歷史水災災點.xlsx")
    record_list = np.array(xl)

    if save:
        record_list = np.save("/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/EMIC_record.npy", record_list)
        print("npy saved.")
    print("==========")
    print("{} EMIC record loaded.".format(len(record_list)))
    print("Type {}".format(type(record_list)))
    print("==========")