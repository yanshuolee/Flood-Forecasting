import re
import random
import os
import math
import argparse
import json

from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.utils.data as torch_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import get_data
import cal_score
from toolkit import get_index, file_path, analyze, early_stopping, chart

# option: original, 3_hours, 6_hours
df = "3_hours"
ES = early_stopping.EarlyStopping(patience=6)
lr = 0.001
iteration = 500
bs = 512

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.lstm = nn.LSTM(3, 64, 2, batch_first=True)
        self.fc_layer_1 = nn.Linear(96, 32)
        self.fc_layer_2 = nn.Linear(32, 16)
        self.fc_layer_3 = nn.Linear(16, 1)
        torch.nn.init.xavier_normal_(self.fc_layer_1.weight)
        torch.nn.init.xavier_normal_(self.fc_layer_2.weight)
        torch.nn.init.xavier_normal_(self.fc_layer_3.weight)
        
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        torch.nn.init.xavier_normal_(self.conv_1.weight)
        torch.nn.init.xavier_normal_(self.conv_2.weight)
        torch.nn.init.xavier_normal_(self.conv_3.weight)

        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, lstm_in, cnn_in, batch):
        h_0 = torch.empty(2, batch, 64)
        c_0 = torch.empty(2, batch, 64)
        nn.init.orthogonal_(c_0)
        nn.init.xavier_normal_(h_0)
        self.h_0 = h_0.to(device)
        self.c_0 = c_0.to(device)

        lstm_out, _ = self.lstm(lstm_in, (self.h_0, self.c_0))

        x = self.conv_1(cnn_in)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv_3(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)

        concat = torch.cat((lstm_out[:,-1,:], x), dim=1)

        fc_1 = self.fc_layer_1(concat)
        fc_1 = nn.ReLU()(fc_1)
        fc_2 = self.fc_layer_2(fc_1)
        fc_2 = nn.ReLU()(fc_2)
        y_pred = self.fc_layer_3(fc_2)

        return y_pred

def train(show_batch=0):
    FP = file_path.get(df)
    trainX, trainY, testX, testY, trainX_map, testX_map = get_data.get(data_folder=df, radius=10, norm=True)
    trainX_map = np.rollaxis(trainX_map, 3, 1)
    testX_map = np.rollaxis(testX_map, 3, 1)

    batch_index = get_index.generate(trainX.shape[0], bs)
    batch_index_testset = get_index.generate(testX.shape[0], bs)
    trainY = trainY.reshape(trainY.shape[0], 1)
    testY = testY.reshape(testY.shape[0], 1)

    model = CRNN().to(device)
    print(model)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, iteration+1):
        
        running_loss = []
        validation_loss = []
        training_parts = int(batch_index.shape[0] * 0.8)

        print("Epoch: {}/{}".format(epoch, iteration))
        for index, batch in enumerate(batch_index[: training_parts]):
            train_data = torch.FloatTensor(trainX[batch]).to(device)
            train_map = torch.FloatTensor(trainX_map[batch]).to(device)
            train_label = torch.FloatTensor(trainY[batch]).to(device)

            y_pred = model(train_data, train_map, bs)
            train_loss = loss(y_pred, train_label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if show_batch:
                print("[batch {}] training loss: {:.4f}".format(index + 1, train_loss.data))
            running_loss.append(train_loss.data)

        for batch in batch_index[training_parts: ]:
            val_data = torch.FloatTensor(trainX[batch]).to(device)
            val_map = torch.FloatTensor(trainX_map[batch]).to(device)
            val_label = torch.FloatTensor(trainY[batch]).to(device)
            vali_y_pred = model(val_data, val_map, bs)
            vali_loss = loss(vali_y_pred, val_label)
            validation_loss.append(vali_loss.data)
        
        print("loss:", "{:.4f}".format(sum(running_loss) / len(running_loss)), "-", "val_loss:", "{:.4f}".format(sum(validation_loss) / len(validation_loss)))

        ES(sum(validation_loss) / len(validation_loss), model)
        if ES.early_stop:
            print("Early Stopping!")
            break

    # load best model
    del model
    torch.cuda.empty_cache()
    model = CRNN().to(device)
    model.load_state_dict(torch.load(ES.best_model_path))

    # calculate RMSE
    scaler = FP["scale_model"]
    scalerGT = FP["scaler_GT"]
    scalerGT = joblib.load(scalerGT)
    scaler = joblib.load(scaler)		

    train_data = torch.FloatTensor(trainX).to(device)
    train_map = torch.FloatTensor(trainX_map).to(device)
    test_data = torch.FloatTensor(testX).to(device)
    test_map = torch.FloatTensor(testX_map).to(device)

    trainPredict = model(train_data, train_map, len(trainX))
    testPredict = model(test_data, test_map, len(testX))

    trainPredict = trainPredict.cpu().detach().numpy()
    testPredict = testPredict.cpu().detach().numpy()

    w1 = scaler.inverse_transform(testX)[:, 2]
    testY = scalerGT.inverse_transform(testY)
    trainY = scalerGT.inverse_transform(trainY)

    # round to 2 decimal
    testPredict = np.round(testPredict, 2)
    trainPredict = np.round(trainPredict, 2)

    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))

    print("Train RMSE: {:.3f}".format(trainScore))
    print("Test RMSE: {:.3f}".format(testScore))

    # Analysing
    json_path = analyze.run(FP, testY, testPredict, w1, FP["analyzeFile"], df)
    return json_path


if __name__ == "__main__":
    json_path = train()
    cal_score.run(show_detail=0)
    # chart.plot(json_path)