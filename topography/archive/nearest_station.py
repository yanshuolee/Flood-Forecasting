##---------------------------------------------------------
## This module is used to calculate nearest station if
## there are more than 1 grd pair to the station.
##
## Author: Y.S.L
##---------------------------------------------------------

import math

def calculate_distance(station_X, station_Y, centroid_X, centroid_Y):
	
	X = (float(station_X) - centroid_X) ** 2
	Y = (float(station_Y) - centroid_Y) ** 2
	return math.sqrt(X+Y)

def get_station_position(station_info_list, id):
	for s in station_info_list:
		if s[0] == id:
			return [s[1], s[2]]

def pair_nearest_station(station_id, grd_ids, station_info, grd_info):
	DISTANCE = []
	
	for grd_id in grd_ids:
		gsp = get_station_position(station_info, station_id)
		dist = calculate_distance(gsp[0],
									gsp[1],
									grd_info[grd_id][1][0],
									grd_info[grd_id][1][1])
		DISTANCE.append(dist)
	
	dist_min = min(DISTANCE)
	for i in range(len(DISTANCE)):
		if DISTANCE[i] == dist_min:
			return [grd_ids[i]]