##---------------------------------------------------------
## Run "get_roi()" with radius parameter to get the dataset
## This module is used for maps of Tainan.
##
## Author: Y.S.L
##---------------------------------------------------------

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from . import station_grd_merge as sgm

MAX_HEIGHT = 0.0
MIN_HEIGHT = 99999.99
GRD_PATH = "/home/csist/YSL/Topography(archive)/data_in_Taiwan/"

station_grd_pair = sgm.merge_station_grd()
print(station_grd_pair)
grd_info = sgm.get_grd()
station_info = sgm.get_station()

def remove_nextline(data_string):
	tmp = data_string.split("\n")
	return float(tmp[0])

def generate_grd_appendix(reader):
	appendix_latitude = {}
	appendix_longitude = {}
	
	row = reader.readline()
	tmp = row.split(" ")
	lat_record = int(tmp[1]) # initialization
	
	while(row is not ""):
		tmp = row.split(" ")
		if lat_record == int(tmp[1]):
			appendix_longitude[int(tmp[0])] = remove_nextline(tmp[2])
		else:
			appendix_latitude[lat_record] = appendix_longitude
			appendix_longitude = {}
			appendix_longitude[int(tmp[0])] = remove_nextline(tmp[2])
			lat_record = int(tmp[1])
		
		row = reader.readline()
	
	return appendix_latitude

def get_station_position(id):
	for i, long, lat in station_info:
		if id == i:
			return (long, lat)
	
	raise NameError("No station found!")

def get_top_left(XY, grd_id):
	X = float(XY[0]) - int(RADIUS * 20)
	Y = float(XY[1]) + int(RADIUS * 20)
	X = int(X)
	Y = int(Y)
	remainder_X = X % 20
	remainder_Y = Y % 20
	if remainder_X is not 0:
		X = X - remainder_X
	if remainder_Y is not 0:
		Y = Y - remainder_Y
	
	return (X, Y)

def get_buttom_right(XY, grd_id):
	X = 20 * RADIUS * 2 + XY[0]
	Y = XY[1] - 20 * RADIUS * 2
	return (int(X), int(Y))

def generate_arr(top_left_position, buttom_right_position, appendix_latitude, grd_id):
	
	lat_arr = []
	for lat in range(top_left_position[1], buttom_right_position[1]-1, -20):
		long_arr = []
		for long in range(top_left_position[0], buttom_right_position[0]+1, 20):
			# print(grd_id, top_left_position, buttom_right_position, "||", grd_info[grd_id][0], "|| station pos: ", )
			if lat not in appendix_latitude:
				print("lat:", grd_id)
				break
			if long not in appendix_latitude[lat]:
				# print("long_id:", grd_id)
				# print(long, lat)
				print("problem here!!!!!", grd_id)
				continue

			long_arr.append(appendix_latitude[lat][long])
		lat_arr.append(long_arr)
	
	return np.array(lat_arr)
	# return lat_arr

DATA_DICT = {}
def is_valid(top_left_position, buttom_right_position, grd_id, appendix_latitude, station_index):
	X1_bool = grd_info[grd_id][0][0][0] < top_left_position[0] < grd_info[grd_id][0][3][0]
	X2_bool = grd_info[grd_id][0][0][0] < buttom_right_position[0] < grd_info[grd_id][0][3][0]
	Y1_bool = grd_info[grd_id][0][0][1] > top_left_position[1] > grd_info[grd_id][0][3][1]
	Y2_bool = grd_info[grd_id][0][0][1] > buttom_right_position[1] > grd_info[grd_id][0][3][1]
	
	
	
	if X1_bool and X2_bool and Y1_bool and Y2_bool:
		datum = generate_arr(top_left_position, buttom_right_position, appendix_latitude, grd_id)
		DATA_DICT[station_index] = datum
		print("Data generated!")
	else:
		print("out of range id: ", grd_id)
		# if not X1_bool:
			# print("top_left_X: ", grd_info[grd_id][0][0][0], " ", top_left_position[0], " ", grd_info[grd_id][0][3][0])
		# if not X2_bool:
			# print("buttom_right_X: ", grd_info[grd_id][0][0][0], " ", buttom_right_position[0], " ", grd_info[grd_id][0][3][0])
		# if not Y1_bool:
			# print("top_left_Y: ", grd_info[grd_id][0][0][1], " ", top_left_position[1], " ", grd_info[grd_id][0][3][1])
		# if not Y2_bool:
			# print("buttom_right_Y: ", grd_info[grd_id][0][0][1], " ", buttom_right_position[1], " ", grd_info[grd_id][0][3][1])
		
		# raise NameError("Out of range! Please select a smaller radius.")
		# return False
	
def generate_roi(station_XY, grd_id, appendix_latitude, station_index):
	top_left_position = get_top_left(station_XY, grd_id)
	buttom_right_position = get_buttom_right(top_left_position, grd_id)
	
	is_valid(top_left_position, buttom_right_position, grd_id, appendix_latitude, station_index)

def get_roi(_radius = 10, normalize=False, norm_type="MinMax"):
	
	global RADIUS
	RADIUS = _radius
	print("Number of data: ", len(station_grd_pair))

	for station_index in station_grd_pair:
		with open(GRD_PATH+station_grd_pair[station_index][0], "r") as grd_reader:
			appendix_latitude = generate_grd_appendix(grd_reader)
		
		station_XY = get_station_position(station_index)
		
		if station_grd_pair[station_index][0] != "94184006dem.grd":
			generate_roi(station_XY, station_grd_pair[station_index][0], appendix_latitude, station_index)
	
	if normalize:
		if norm_type == "Standard":
			scalar = StandardScaler()
		if norm_type == "MinMax":
			scalar = MinMaxScaler()
		
		for id in DATA_DICT:
			unnormalized_data = DATA_DICT[id]
			DATA_DICT[id] = scalar.fit_transform(unnormalized_data)
	
	return DATA_DICT
