import numpy as np
import os
import json

def read_nparray(filePath):
    data = np.load(filePath)
    return data

def write_json(x, fileName):
    with open(fileName, "w") as f:
        data = json.dumps(x)
        json.dump(data, f)

def run(FP, testY, testPredict, w1, fileName, df, remove_list=None):
    with open(fileName, "w", encoding="utf-8") as writer:
        result_dict = {}
        writer.write(
            "STD\t" + "GT" + "\t" + "Predicted" + "\t" + "WarningLine 1" + "\tGT_result\t" + "Predicted Result" + "\n")
        GT_result = 0
        Pred_result = 0
        testY = read_nparray(FP["testYWaterInfo"])

        if remove_list is not None:
            testY = np.delete(testY, remove_list, axis=0)
            
        for i in range(0, len(testY)):
            
            pred_WL = round(testPredict[i][0] + float(testY[i][2]), 1)
            
            if float(testY[i][1]) > float(w1[i][1]):
                GT_result = 1
            else:
                GT_result = 0

            if pred_WL > float(w1[i][1]):
                Pred_result = 1
            else:
                Pred_result = 0

            writer.write(str(testY[i][0]) + "\t" + str(testY[i][1]) + "\t" + str(
                pred_WL) + "\t" + str(w1[i][1]) + "\t" + str(GT_result) + "\t" + str(
                Pred_result) + "\n")

            Std= str(testY[i][0])

            if Std not in result_dict:
                result_dict[Std] = {}
                result_dict[Std]["WarnLine"] = w1[i][1]

            time = (testY[i][3]).strftime("%Y-%m-%d")

            if time not in result_dict[Std]:
                result_dict[Std][time] = {}
                
            time1 =(testY[i][3]).strftime("%Y-%m-%d %H:%M:%S")
            result_dict[Std][time][time1] = (testY[i][1], pred_WL)

    write_json(result_dict, os.path.join("exp", "Ori_model", "result_{}.json".format(df)))

    return os.path.join("exp", "Ori_model", "result_{}.json".format(df))