#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import measure
import torch
import pandas as pd
from trainer import setup_seed, Divide, TrainValidTest
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda_gpu = torch.cuda.is_available()
class FCLSTM(nn.Module):
    name = 'FC-LSTM'
    def __init__(self, input_num, output_num, hidden_num):
        super(FCLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_num, hidden_size=hidden_num, num_layers=1,bias=True, batch_first=True)
        self.reg = nn.Linear(in_features=hidden_num, out_features=output_num)
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.reg(output[:, -1, :])
        return output

def preprocess_data(x, y, feature_num, seq_len, pre_len):
    nodes = x.shape[1]
    if feature_num == 1:
        x_ones, x_min, x_max = measure.Norm(np.array(torch.tensor(x).view(-1), dtype=float), dim=1)
        source_x = torch.tensor(x_ones).view(-1, nodes)
    else:
        x_ones, x_min, x_max = measure.Norm(np.array(torch.tensor(x).view(-1, feature_num), dtype=float), dim=2)
        source_x = torch.tensor(x_ones).view(-1, nodes, feature_num)
    global y_min
    global y_max
    y_ones, y_min, y_max = measure.Norm(np.array(torch.tensor(y).view(-1), dtype=float), dim=1)

    source_y = torch.tensor(y_ones).view(-1, nodes)

    hist_x, fore_y = Divide(source_x, source_y, seq_len, pre_len)
    return TrainValidTest(hist_x, fore_y, valid=0.05)
    # train_x, train_y, test_x, test_y = Main.TrainValidTest(hist_x, fore_y)


def HA_forecasr(test_x, test_y, pre_len):
    for i in range(len(test_x)):
        a = test_x[i]
        a1 = torch.mean(a, dim=1)
        for j in range(pre_len):
            if j == 0:
                temp = a1.view(-1, 1)
            else:
                temp = torch.cat((temp, a1.view(-1, 1)), dim=1)
        temp = temp.unsqueeze(0)
        if i == 0:
            result = temp
        else:
            result = torch.cat((result, temp), dim=0)
    print("result.shape: ", result.shape)
    result = result * (y_max - y_min) + y_min
    test_y = test_y * (y_max - y_min) + y_min
    result1 = torch.tensor(result, dtype=torch.float32)
    print("RMSE: ", measure.GetRMSE(test_y, result1))
    print("MAP: ", measure.GetMAP(test_y, result1))
    print("SKILL: ", measure.Skill(test_y, result1))
    print("MAPE: ", measure.GetMAPE(test_y, result))
    print("R2: ", measure.R_square(test_y, result))


def SVR_forecast(train_x, train_y, test_x, test_y, pre_len):
    nodes = test_y.shape[1]
    print(nodes)
    result = []
    for i in range(nodes):
        a_X = np.array(train_x[:, i, :])
        print(a_X.shape)
        a_Y = np.array(train_y[:, i, 0])
        t_X = np.array(test_x[:, i, :])
        print(t_X.shape)
        svr_model = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf', 'sigmoid'), "C": np.logspace(-3, 3, 7),
                                                    "gamma": np.logspace(-3, 3, 7)}, cv=5)
        # svr_model = SVR('rbf')
        svr_model.fit(a_X, a_Y)
        for i in range(pre_len):
            if i == 0:
                pre = svr_model.predict(t_X)
                Predict = pre.reshape(-1, 1)
                # print("Predict: ", Predict.shape)
            else:
                t_X = np.concatenate((t_X[:, 1:], pre.reshape(-1, 1)), axis=1)
                pre = svr_model.predict(t_X)
                Predict = np.concatenate((Predict, pre.reshape(-1, 1)), axis=1)
        print(Predict.shape)
        result.append(Predict)
    result = np.array(result)
    result = torch.tensor(result).permute(1, 0, 2) * (y_max - y_min) + y_min
    test_y = test_y * (y_max - y_min) + y_min
    print("RMSE: ", measure.GetRMSE(test_y, result))
    print("MAP: ", measure.GetMAP(test_y, result))
    print("SKILL: ", measure.Skill(test_y, result))
    print("MAPE: ", measure.GetMAPE(test_y, result))
    print("R2: ", measure.R_square(test_y, result))
    print(result.shape)


def ARIMA_forecast(data, order=[2, 0, 0]):
    dta = pd.DataFrame(data)
    rng = pd.date_range('1/1/2015', periods=5848, freq='3h')
    dta.index = pd.DatetimeIndex(rng)
    RMSE = []
    MAP = []
    MAPE = []
    R2 = []
    for i in range(data.shape[1]):
        ts = dta.iloc[:, i]
        ts_log = np.log(ts)
        ts_log = np.array(ts_log, dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = pd.DatetimeIndex(rng)
        model = ARIMA(ts_log, order=order)
        properModel = model.fit()
        predict_ts = properModel.predict(4971, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        RMSE.append(measure.GetRMSE(torch.from_numpy(ts.values), torch.from_numpy(log_recover.values)))
        MAP.append(measure.GetMAP(torch.from_numpy(ts.values), torch.from_numpy(log_recover.values)))
        MAPE.append(measure.GetMAPE(torch.from_numpy(ts.values), torch.from_numpy(log_recover.values)))
        R2.append(measure.R_square(torch.from_numpy(ts.values), torch.from_numpy(log_recover.values)))
    print("RMSE: ", np.mean(RMSE))
    print('MAP: ', np.mean(MAP))
    print('MAPE: ', np.mean(MAPE))
    print('R2: ', np.mean(R2))


def LSTM_forecast(train_x, train_y, valid_x, valid_y, test_x, test_y):
    nodes = test_y.shape[1]
    print(nodes)
    epoch = 200
    batch = 200
    setup_seed(20)
    criterion = nn.MSELoss()
    for i in range(nodes):
        # 得到节点180的训练模型
        train_loader = DataLoader(dataset=TensorDataset(train_x[:, i, :, :], train_y[:, i, :]), batch_size=batch,
                                  shuffle=True)
        lstm_model = FCLSTM(9, 8, 48).to(device)
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
        for e in range(epoch):
            for _, data in enumerate(train_loader):
                inputs, target = data
                if cuda_gpu:
                    inputs, target = Variable(inputs).to(device), Variable(target).to(device)
                else:
                    inputs, target = Variable(inputs), Variable(target)
                y_hat = lstm_model(inputs)
                loss = criterion(y_hat, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (e + 1) % 10 == 0:
                # print(device)
                valid = valid_x[:, i, :, :].to(device)

                print('Epoch [{}/{}], Loss:{:.4f}'.format(e + 1, epoch, loss.item()))
                valid_outputs = lstm_model(valid)
                valid_outputs = valid_outputs.cpu()
                valid_outputs = valid_outputs * (y_max - y_min) + y_min
                Valid_y = valid_y * (y_max - y_min) + y_min
                rmse = measure.GetRMSE(valid_outputs, Valid_y[:, i, :])
                print("valid RMSE:", rmse)

        torch.save(lstm_model, 'FC_LSTM.pkl')
        print('over')
        test = test_x[:, i, :, :].to(device)
        test_outputs = lstm_model(test)
        if i == 0:
            result = test_outputs
        else:
            result = torch.cat((result, test_outputs), dim=1)
    result = result.cpu() * (y_max - y_min) + y_min
    result = result.view(test_y.shape)
    test_y = test_y * (y_max - y_min) + y_min
    print("RMSE: ", measure.GetRMSE(test_y, result))
    print("MAP: ", measure.GetMAP(test_y, result))
    print("SKILL: ", measure.Skill(test_y, result))
    print("MAPE: ", measure.GetMAPE(test_y, result))
    print("R2: ", measure.R_square(test_y, result))


if __name__ == "__main__":
    dataset = np.load("../data/DataSet.npy")
    x = dataset[:5848, :, :]
    y = dataset[:5848, :, -1]
    train_x, train_y, valid_x, valid_y, test_x, test_y = preprocess_data(x, y, feature_num=9, seq_len=16, pre_len=8)
    start = time.time()
    HA_forecasr(test_x[:, :, :, -1], test_y, 8)
    SVR_forecast(train_x[:, :, :, -1], train_y, test_x[:, :, :, -1], test_y, pre_len=8)
    ARIMA_forecast(x)
    LSTM_forecast(train_x, train_y, valid_x, valid_y, test_x, test_y)
    end = time.time()
    print('time cost: ', end - start)

