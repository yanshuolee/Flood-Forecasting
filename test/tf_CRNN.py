import re
import random
import os
import math
import argparse
import json

from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

import get_data
# option: original, 3_hours, 6_hours
df = "3_hours"

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

def write_json(x,fileName):
    with open(fileName,"w") as f:
        data = json.dumps(x)
        json.dump(data, f)

def getFilePath():
    FP = {}
    FP["trainXFp"] = os.path.join("exp", "train", df, "nor_train_merge_WL_w1.npy")
    FP["trainYFp"] = os.path.join("exp", "train", df, "train_gt_merge_WL_w1.npy")
    FP["testXFp"] = os.path.join("exp", "test", df, "nor_test_merge_WL_w1.npy")
    FP["testYFp"] = os.path.join("exp", "test", df, "test_gt_merge_WL_w1.npy")
    FP["testYWaterInfo"] = os.path.join("exp", "test", df, "test_gt_merge_WL_info.npy")

    # FP["testXFp"] = os.path.join("exp", "test", "bal","bal_nor_test_merge_WL_w1.npy")
    # FP["testYFp"] = os.path.join("exp", "test","bal","bal_test_gt_merge_WL_w1.npy")

    FP["scale_model"] = os.path.join("exp", "scaler", "mymodel.pkl")
    FP["scaler_GT"] = os.path.join("exp", "scaler", "GT", "GTmodel.pkl")

    FP["analyzeFile"] = os.path.join("exp", "Ori_model", "analyze_ori.txt")
    # FP["modelFp"] = os.path.join("exp", "Ori_model", "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
    FP["modelFp"] = os.path.join("exp", "Ori_model", "weights.hdf5")
    FP["logFp"] = os.path.join("exp", "Ori_model", "log.txt")
    return FP

def read_nparray(filePath):
    print("File path: ", filePath)
    data = np.load(filePath)
    return data

def generate_batch_index(data_length, batch_size, shuffle=False):
	row = data_length//batch_size
	row_remain = data_length%batch_size
	col = batch_size
	total_length = row*col
	arr = np.arange(total_length)
	if row_remain:
		arr_adder = np.arange(total_length+row_remain-1, total_length+row_remain-1-batch_size, -1)
	
	if shuffle:
		arr = arr.reshape((row, col))
		concat = np.append(arr, arr_adder, axis=0)
		np.random.shuffle(concat)
		concat = concat.reshape((row, col))
	else:
		arr = arr.reshape((row, col))		
		concat = np.append(arr, [arr_adder], axis=0)
	
	return concat

'''
# 3-hour model
def three_hr_model_structure(timestep, dimension, row, col):

	model_lstm_in = Input(shape=(timestep, dimension))
	model_lstm_out = LSTM(units=512, recurrent_dropout=0.2)(model_lstm_in)
	# model_lstm = Dense(units=32)(model_lstm)
	# model_lstm_out = Dense(units=16)(model_lstm)

	model_cnn_in = Input(shape=(row, col, 1))
	model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn_in)
	model_cnn = MaxPooling2D(pool_size=2)(model_cnn)
	model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn)
	model_cnn = MaxPooling2D(pool_size=2)(model_cnn)
	model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn)
	model_cnn_out = Flatten()(model_cnn)

	concatenate_model = concatenate([model_lstm_out, model_cnn_out])

	concatenate_model = Dense(10, activation="relu")(concatenate_model)
	out = Dense(units=1, activation='tanh')(concatenate_model)

	merged_model = Model([model_lstm_in, model_cnn_in], out)

	print(merged_model.summary())

	return merged_model

# 6-hour model
def six_hr_model_structure(timestep, dimension, row, col):

	model_lstm_in = Input(shape=(timestep, dimension))
	model_lstm_out = LSTM(units=512)(model_lstm_in)
	# model_lstm_out = Dense(units=16, activation="relu")(model_lstm)

	model_cnn_in = Input(shape=(row, col, 1))
	model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn_in)
	model_cnn = MaxPooling2D(pool_size=2)(model_cnn)
	model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn)
	model_cnn = MaxPooling2D(pool_size=2)(model_cnn)
	model_cnn = Conv2D(filters=10, kernel_size=3, activation="relu")(model_cnn)
	model_cnn_out = Flatten()(model_cnn)

	concatenate_model = concatenate([model_lstm_out, model_cnn_out])

	concatenate_model = Dense(10, activation="relu")(concatenate_model)
	out = Dense(units=1, activation='tanh')(concatenate_model)

	merged_model = Model([model_lstm_in, model_cnn_in], out)

	print(merged_model.summary())

	return merged_model

def get_model(name, timestep, dimension, row, col):
	if name == "original":
		model = original_model_structure(timestep, dimension, row, col)
	elif name == "3_hours":
		model = three_hr_model_structure(timestep, dimension, row, col)
	elif name == "6_hours":
		model = six_hr_model_structure(timestep, dimension, row, col)
	print(name, "model has been selected!")
	return model
'''

def get_state_variables(batch_size, cell):
	state_variables = []
	state_c, state_h = cell.zero_state(batch_size, tf.float32)
	state_variables.append(tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, name="c_state"),
														tf.Variable(state_h, name="h_state")))
	return tf.contrib.rnn.LSTMStateTuple(tf.Variable(state_c, trainable=False), tf.Variable(state_h, trainable=False))
	# cell_state = tf.placeholder(tf.float32, [bs, 64], name="cell_state")
	# hidden_state = tf.placeholder(tf.float32, [bs, 64], name="hidden_state")
	# init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
	# return init_state

lr = 0.001
iteration = 100
bs = 1000
def train():
	FP = getFilePath()
	trainX, trainY, testX, testY, trainX_map, testX_map = get_data.get(data_folder=df, radius=10, norm=True)
	trainY = trainY.reshape(trainY.shape[0], 1)
	batch_index = generate_batch_index(trainX.shape[0], bs)
	testY = testY.reshape(testY.shape[0], 1)

	graph = tf.Graph()
	with graph.as_default():
		data = tf.placeholder(dtype=tf.float32, shape=(None, 3, 3), name="lstm_input")
		labels = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="labels")

		rnn_layers = [tf.nn.rnn_cell.LSTMCell(num_units=size) for size in [64, 64]]
		rnn_cells = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
		states = get_state_variables(bs, rnn_cells)
		lstm, state = tf.nn.dynamic_rnn(rnn_cells, data, initial_state=states, dtype=tf.float32)

		outputs = tf.contrib.layers.fully_connected(inputs=state[1].h, num_outputs=32, activation_fn=tf.nn.relu)
		outputs = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=16, activation_fn=tf.nn.relu)
		outputs = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=1, activation_fn=tf.nn.tanh)
		
		cost = tf.reduce_mean(tf.losses.mean_squared_error(labels, outputs))
		optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

	with tf.Session(graph=graph) as sess:
		sess.run(tf.global_variables_initializer())
		writer = open("log.txt", "w+")
		saver = tf.train.Saver()

		for epoch in range(1, iteration+1):
			
			training_parts = int(batch_index.shape[0] * 0.8)
			for batch in batch_index[: training_parts]:
				loss, opt = sess.run([cost, optimizer], feed_dict={data: trainX[batch], labels: trainY[batch]})

			val_loss, val_opt, _state = sess.run([cost, optimizer, state], feed_dict={data: trainX[training_parts:], labels: trainY[training_parts:]})
			print("Epoch: {}/{}".format(epoch, iteration))
			print("loss:", "{:.4f}".format(loss), "-", "val_loss:", "{:.4f}".format(val_loss))
			if epoch%10 ==0:
				saver.save(sess, "/weights", global_step=epoch)

		# calculate RMSE
		scaler = FP["scale_model"]
		scalerGT = FP["scaler_GT"]
		trainPredict = sess.run(outputs, feed_dict={data: trainX})
		testPredict = sess.run(outputs, feed_dict={data: testX})

		scalerGT = joblib.load(scalerGT)
		scaler = joblib.load(scaler)

		trainPredict = scalerGT.inverse_transform(trainPredict)
		testPredict = scalerGT.inverse_transform(testPredict)
		print(testPredict.shape)

		w1 = scaler.inverse_transform(testX)[:, 2]

		testY = scalerGT.inverse_transform(testY)
		print(testY.shape)
		trainY = scalerGT.inverse_transform(trainY)

		testScore = math.sqrt(mean_squared_error(testY, testPredict))
		trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))

		print("Train RMSE: " + str(trainScore))
		print("Test RMSE: " + str(testScore))


		# Analysing
		analyze(FP, testY, testPredict, w1, FP["analyzeFile"])

def analyze(FP, testY, testPredict, w1, fileName):

	with open(fileName, "w", encoding="utf-8") as writer:
		result_dict = {}
		writer.write(
			"STD\t" + "GT" + "\t" + "Predicted" + "\t" + "WarningLine 1" + "\tGT_result\t" + "Predicted Result" + "\n")
		GT_result = 0
		Pred_result = 0
		testY = read_nparray(FP["testYWaterInfo"])
		print("testY.shape = ", testY.shape)
		print("testY[0]: ", testY[0])
		for i in range(0, len(testY)):
			if float(testY[i][1]) > float(w1[i][1]):
				GT_result = 1
			else:
				GT_result = 0

			if (testPredict[i][0] + float(testY[i][2])) > float(w1[i][1]):
				Pred_result = 1
			else:
				Pred_result = 0

			writer.write(str(testY[i][0]) + "\t" + str(testY[i][1]) + "\t" + str(
				testPredict[i][0] + float(testY[i][2])) + "\t" + str(w1[i][1]) + "\t" + str(GT_result) + "\t" + str(
				Pred_result) + "\n")
			Std= str(testY[i][0])
			if Std not in result_dict:
				result_dict[Std] = {}
				result_dict[Std]["WarnLine"] = w1[i][2]
			time = (testY[i][3]).strftime("%Y-%m-%d")
			if time not in result_dict[Std]:
				result_dict[Std][time] = {}
			time1 =(testY[i][3]).strftime("%Y-%m-%d %H:%M:%S")
			result_dict[Std][time][time1] = (testY[i][1], testPredict[i][0] + float(testY[i][2]))
	write_json(result_dict, os.path.join("result", "result_test_ver.json"))

if __name__ == "__main__":
	train()