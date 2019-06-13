import numpy as np
import jieba
import pickle
import datetime
from toolkit import tools
threshold = 300
agency_path = "/mnt/HDD_1/yanshuo/Elderly_Welfare_Agency_2017/Tainan_agency_with_wl_rain_station.npy"
flooding_rec_path = "/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/EMIC_record.npy"

agency = tools.read_nparray(agency_path)
flooding_rec = tools.read_nparray(flooding_rec_path, allow_pickle=True)

def flooding_near_agency():
    # Distance btw agency and EMIC within threshold
    count_rec = 0
    flooding_near_agency_keys = []
    flooding_near_agency_values = []
    INDEX = []

    for i, record in enumerate(flooding_rec):
        for ag in agency:
            distance = tools.cal_dist(record[6], record[7], float(ag[18]), float(ag[19]))
            if distance <= threshold:
                flooding_near_agency_keys.append(str(ag[2]))
                flooding_near_agency_values.append([ag[23], ag[24], record[0], record[6], record[7], record[11]])
                INDEX.append(i)
                count_rec = count_rec + 1

    print("{}/{} EMIC record within {} meter.".format(count_rec, len(flooding_rec), threshold))

    return flooding_near_agency_keys, flooding_near_agency_values, INDEX

def labelling(keys, values, filename, index):
    # Determine if the description contains words like "cm" or "m"
    # description: array index [11]

    labeled_keys = []
    labeled_values = []
    labeled_index = []
    unlabeled_keys = []
    unlabeled_values = []
    unlabeled_index = []

    for key, record, ii in zip(keys, values, index):
        words = jieba.cut(str(record[-1]))
        words = list(words)
        
        if "公分" in words:
            labeled_keys.append(key)
            labeled_values.append(record)
            labeled_index.append(ii)
        elif "公尺" in words:
            labeled_keys.append(key)
            labeled_values.append(record)
            labeled_index.append(ii)
        else:
            unlabeled_keys.append(key)
            unlabeled_values.append(record)
            unlabeled_index.append(ii)
    
    print("")

    # _labeled_keys, _labeled_values,_unlabeled_keys, _unlabeled_values = tools.human_labeling(labeled_keys, labeled_values, unlabeled_keys, unlabeled_values)

    # while True:
    #     inputs = input("Save? Yes(1)/No(0): ")
    #     inputs = int(inputs)
    #     if inputs == 0 or inputs == 1: break
    
    # if inputs == 1:
    #     save_path = "/mnt/HDD_1/yanshuo/EMIC2016-2019台南市歷史水災災點/"
    #     np.save(save_path+filename+"_labeled_key.npy", _labeled_keys)
    #     np.save(save_path+filename+"_labeled_value.npy", _labeled_values)
    #     np.save(save_path+filename+"_unlabeled_key.npy", _unlabeled_keys)
    #     np.save(save_path+filename+"_unlabeled_value.npy", _unlabeled_values)
    #     print("Saved!")

    # print("labeled: {}, unlabeled: {}, {} in total.".format(len(_labeled_values), len(_unlabeled_values), len(keys)))

if __name__ == "__main__":
    flooding_near_agency_keys, flooding_near_agency_values, index = flooding_near_agency()
    labelling(flooding_near_agency_keys, flooding_near_agency_values, "elder", index)

print("t")