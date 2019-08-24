import json
import os
import itertools
import pylab as plt
import matplotlib
from datetime import datetime
import numpy as np

def read_json(fileName):
    with open(fileName) as json_file:
        json_str = json.load(json_file)
        data = json.loads(json_str)
    return data

def plot(fileName):

    start = datetime.strptime('2015-05-01', '%Y-%m-%d')
    end = datetime.strptime('2015-11-30', '%Y-%m-%d')

    result = read_json(fileName)
    saved_directory = os.path.join("exp", "Ori_model", "chart_png")
    verbose_dict = {}

    if not os.path.exists(saved_directory):
        os.makedirs(saved_directory)

    for place in result.keys():
        pred_dict = {}
        gt_dict = {}
        date_time = []
        pred = []
        gt = []

        for date in result[place].keys():
            if date == 'WarnLine':
                wl = result[place][date]
            else:
                for time in result[place][date].keys():
                    current_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                    if current_time > start and current_time < end:
                        pred_dict[current_time] = result[place][date][time][1]
                        gt_dict[current_time] = result[place][date][time][0]

        for date in sorted(pred_dict.keys()):
            date_time.append(date)
            pred.append(pred_dict[date])

        for date in sorted(gt_dict.keys()):
            gt.append(gt_dict[date])

        plt.figure(figsize=(10,5))
        
        plot_pred, = plt.plot(date_time, pred)
        plot_gt, = plt.plot(date_time, gt)
        plt.axhline(y=wl, color='red')

        # plt.title(place)
        plt.xticks(rotation=35)
        plt.legend(handles = [plot_gt, plot_pred], labels = ["Ground Truth", "Prediction"], loc = 'best' )
        plt.savefig("{}/{}.png".format(saved_directory, place))
        plt.clf()
        verbose_dict[place] = [pred, gt]

    plt.close()

    print("Plot finished!")
