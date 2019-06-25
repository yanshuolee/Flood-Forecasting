import numpy as np

PATH = "/home/csist/YSL/Flood/topography/tainan/"

def create_matrix():
	with open(PATH+"all_tainan_edited.txt", "r") as data_reader:
		row = data_reader.readline()
		data = np.matrix(row)
		print(data.shape)
		np.save("./tainan.npy", data)

