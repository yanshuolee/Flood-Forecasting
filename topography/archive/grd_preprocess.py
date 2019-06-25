##---------------------------------------------------------
## Import this module and run "get_grd_info()" to retreive
## GRD dictionary.
##
## Author: Y.S.L
##---------------------------------------------------------

import numpy as np
import os

GRD_PATH = "/home/csist/YSL/Flood/topography/data_in_Tainan/"
FILE_NAME_INDEX = []
GRD_INFO = {}

def get_grd_info():
	open_manifest()
	make_grd_info()
	if not os.path.isfile("file_name_index.npy"):
		save_file_name_index()
	
	return GRD_INFO

def save_file_name_index():
	np.save("file_name_index.npy", np.array(FILE_NAME_INDEX))
	print("file_name_index.npy saved!")

def remove_nextline(data_string):
	tmp = data_string.split("\n")
	return tmp[0]

def open_manifest():
	with open(GRD_PATH+"manifest.csv", "r") as file:
		
		data = file.readline() # get rid of the header
		data_1 = file.readline()
		data_2 = file.readline()
		while(data_1 is not ""):
			tmp_1 = data_1.split(",,\n")
			tmp_2 = data_2.split(",,\n")
			FILE_NAME_INDEX.append((tmp_1[0].split(" ")[0], tmp_2[0].split(" ")[0]))
			data_1 = file.readline()
			data_2 = file.readline()

def make_grd_info():
	for grd, hdr in FILE_NAME_INDEX:
		
		header_reader = open(GRD_PATH+hdr, encoding='utf8', errors='ignore')
		
		top_left, top_right, buttom_left, buttom_right = get_corner(header_reader)
		centroid_X, centroid_Y = calculate_centroid(top_left)
		
		GRD_INFO[grd] = [(top_left, top_right, buttom_left, buttom_right),
						(centroid_X, centroid_Y),
						(stride_X, stride_Y)
						]
		
		
	header_reader.close()

def calculate_centroid(top_left):
	return_value = (top_left[0]+HALF_STRIDE[0], top_left[1]+HALF_STRIDE[1])
	return return_value


def get_corner(header_reader):
	header_data = header_reader.readlines()
	
	global stride_X
	global stride_Y
	stride_X = int(remove_nextline(header_data[8]))
	stride_Y = int(remove_nextline(header_data[9]))
	global HALF_STRIDE
	HALF_STRIDE = (stride_X//2, stride_Y//2)
	
	buttom_left_X = int(remove_nextline(header_data[10]))
	buttom_left_Y = int(remove_nextline(header_data[11]))
	buttom_left = (buttom_left_X, buttom_left_Y)
	
	top_right_X = buttom_left_X + stride_X * (int(remove_nextline(header_data[5]))-1)
	top_right_Y = buttom_left_Y + stride_Y * (int(remove_nextline(header_data[6]))-1)
	top_right = (top_right_X, top_right_Y)
		
	buttom_right = (top_right_X, buttom_left_Y)
	
	top_left = (buttom_left_X, top_right_Y)
	
	return top_left, top_right, buttom_left, buttom_right
