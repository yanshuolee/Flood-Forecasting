##---------------------------------------------------------
## Import this module and run "merge_station_grd()" to
## retreive station-grd pair.
##
## Author: Y.S.L
##---------------------------------------------------------

import numpy as np
import grd_preprocess as gp
import nearest_station as ns

WL_STATION_COORDINATE = "/home/csist/YSL/Flood/Data/WL_station_coordinate.npy"

OVERLAP_STATION_ID = []
STATION_PAIRED_COLLECTION = {}

def get_station():
	station_info = np.load(WL_STATION_COORDINATE)
	return station_info

def get_grd_id(grd_dict):
	
	for i in grd_dict.keys():
		id = i
	return id

def get_grd():
	grd_info = gp.get_grd_info()
	return grd_info

def is_inside(station_X, station_Y, grd_left_top, grd_right_buttom):
	
	if grd_left_top[0] < float(station_X) < grd_right_buttom[0]:
		if grd_left_top[1] > float(station_Y) > grd_right_buttom[1]:
			return True
		else:
			return False
	else:
		return False

def check_station_paired_collection(station_info, grd_info):
	alone_station = False
	alone_station_list = []
	
	for i, _, _ in station_info:
		if i not in list(STATION_PAIRED_COLLECTION.keys()):
			alone_station = True
			alone_station_list.append(i)
	
	# for i in list(STATION_PAIRED_COLLECTION.keys()):
		# if(len(STATION_PAIRED_COLLECTION[i]) > 1):
			# pns = ns.pair_nearest_station(i, STATION_PAIRED_COLLECTION[i], station_info, grd_info)
			# STATION_PAIRED_COLLECTION[i] = pns
	
	for i in OVERLAP_STATION_ID:
		pns = ns.pair_nearest_station(i, STATION_PAIRED_COLLECTION[i], station_info, grd_info)
		STATION_PAIRED_COLLECTION[i] = pns
	
	if alone_station:
		print("There are alone station, and it has been remove!")
		print("Alone station: ", alone_station_list)

def merge_station_grd():
	station_info = get_station()
	print(station_info)
	grd_info = get_grd()
	
	for station in station_info:
		for grd in grd_info:
			
			if is_inside(station[1], station[2], grd_info[grd][0][0], grd_info[grd][0][3]):
				if station[0] in STATION_PAIRED_COLLECTION:
					if station[0] not in OVERLAP_STATION_ID:
						OVERLAP_STATION_ID.append(station[0])
					STATION_PAIRED_COLLECTION[station[0]].append(grd)
				else:
					STATION_PAIRED_COLLECTION[station[0]] = [grd]
			# else:
				# print("station position: ", station, "top-left: ", grd_info[grd][0][0], "buttom-right: ", grd_info[grd][0][3])
	
	
	print("Overlap station id:", OVERLAP_STATION_ID)
	check_station_paired_collection(station_info, grd_info)
	
	return STATION_PAIRED_COLLECTION
