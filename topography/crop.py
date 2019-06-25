##---------------------------------------------------------
## Run "main()" to get the dictionary of ROI.
## This module is used for the whole map of Tainan.
##
## Author: Y.S.L
##---------------------------------------------------------

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PATH = "topography/tainan/tainan.npy"
HEIGHT_MATRIX = np.load(PATH)
BUTTOM_LEFT_X = 148590
BUTTOM_LEFT_Y = 2530590
STRIDE = 20
N_ROWS = HEIGHT_MATRIX.shape[0]
N_COLS = HEIGHT_MATRIX.shape[1]
DATA_DICT = {}

def get_right_top():
	X = (N_COLS-1) * STRIDE + BUTTOM_LEFT_X
	Y = (N_ROWS-1) * STRIDE + BUTTOM_LEFT_Y
	return (X, Y)

TOP_RIGHT = get_right_top()

def get_TL_BR():
	TL = (BUTTOM_LEFT_X, TOP_RIGHT[1])
	BR = (TOP_RIGHT[0], BUTTOM_LEFT_Y)
	return TL, BR

TL_position, _ = get_TL_BR()

def get_roi_TL_BR(XY):
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
	
	X_1 = 20 * RADIUS * 2 + X
	Y_1 = Y - 20 * RADIUS * 2
	return (X, Y, X_1, Y_1)

def is_inside(X, Y):
	if BUTTOM_LEFT_X < float(X) < TOP_RIGHT[0]:
		if BUTTOM_LEFT_Y < float(Y) < TOP_RIGHT[1]:
			return True
		else:
			return False
	else:
		return False

def is_valid(TL_BR_position):
	TL_bool = is_inside(TL_BR_position[0], TL_BR_position[1])
	BR_bool = is_inside(TL_BR_position[2], TL_BR_position[3])
	
	if TL_bool and BR_bool:
		return True
	else:
		raise NameError("ROI out of range! Choose smaller range!")

def map_to_matrix(X, Y):
	new_X = abs(X-TL_position[0]) // STRIDE
	new_Y = abs(Y-TL_position[1]) // STRIDE
	return new_X, new_Y

def generate_arr(TL_BR_position):
	# print("unmap:", TL_BR_position)
	TL_X, TL_Y = map_to_matrix(TL_BR_position[0], TL_BR_position[1])
	BR_X, BR_Y = map_to_matrix(TL_BR_position[2], TL_BR_position[3])
	# print("mapped: ", TL_X, TL_Y, BR_X, BR_Y)
	roi_matrix = HEIGHT_MATRIX[TL_Y:BR_Y+1, TL_X:BR_X+1]
	return roi_matrix

def no_data(matrix):
	type_1 = -9999 in matrix
	type_2 = -999 in matrix
	if type_1:
		if type_2:
			for i in range(matrix.shape[0]):
				for j in range(matrix.shape[1]):
					if matrix[i][j] == -9999:
						matrix[i][j] = 0
					if matrix[i][j] == -999:
						matrix[i][j] = 0
		else:
			for i in range(matrix.shape[0]):
				for j in range(matrix.shape[1]):
					if matrix[i][j] == -9999:
						matrix[i][j] = 0
	
	return matrix

def foo(X, Y):
	global RADIUS
	RADIUS = 10
	if is_inside(X, Y):
		TL_BR_position = get_roi_TL_BR((X, Y))
		if is_valid(TL_BR_position):
			roi_matrix = generate_arr(TL_BR_position)
			return no_data(roi_matrix)
	else:
		print("BL: ", (BUTTOM_LEFT_X, BUTTOM_LEFT_Y), "TR: ", TOP_RIGHT)
		raise ArithmeticError("out of range!")