#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as mp
import sklearn.metrics as sm
import sklearn.ensemble as se  # 集合算法模块
import sklearn.utils as su
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
import model_fs
import time
import measure

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
attribute_name = np.load("../data/attribute_name.npy")
print(attribute_name)
attribute = np.concatenate((attribute_name[0:3], attribute_name[4:10],
                            attribute_name[12:], ['pm2.5']), axis=0)
cuda_gpu = torch.cuda.is_available()


# print("use gpu:", cuda_gpu)
# print(attribute.shape)
def ratioforscore(scorelist):
    sum = np.sum(scorelist)
    ratio = scorelist / sum
    return ratio


def ImportantInRF(x, y):
    print(x.shape)
    feature_num = x.shape[-1]
    nodes = x.shape[1]
    print(feature_num)
    x, _, _ = measure.Norm(np.array(torch.tensor(x).view(-1, feature_num)), dim=2)
    x = x.reshape(-1, nodes, feature_num)
    x, y = su.shuffle(x, y, random_state=0)
    train_size = int(len(x) * 0.9)
    score = None
    importances = None
    for i in range(x.shape[1]):
        start = time.time()
        x_city = x[:, i, :]
        y_city = y[:, i]
        train_x, test_x, train_y, test_y = x_city[:train_size], x_city[train_size:], \
                                           y_city[:train_size], y_city[train_size:]
        model = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=3)
        model.fit(train_x, train_y)
        # 模型测试
        pred_test_y = model.predict(test_x)
        if i == 0:
            score = sm.r2_score(test_y, pred_test_y)
            importances = model.feature_importances_
        else:
            importances += model.feature_importances_
            score += sm.r2_score(test_y, pred_test_y)
        end = time.time()
        print('time: ', end - start)
    importances = importances / x.shape[1]
    score = score / x.shape[1]
    print('score: ', score)
    sorted_indexes = importances.argsort()[::-1]
    print(importances)
    print(attribute[sorted_indexes])


# his_step, fore_step
def Divide(his_step, fore_step, x, y):
    # hist_x = (batch_size, step, feature)
    # fore_y = (batch_size, step)
    hist_x = np.zeros((x.shape[0] - his_step - fore_step + 1, his_step, x.shape[1]))
    fore_y = np.zeros((y.shape[0] - his_step - fore_step + 1, fore_step))
    for i in range(x.shape[0] - his_step - fore_step + 1):
        for j in range(his_step):
            hist_x[i, j, :] = x[i + j, :]
        for j in range(fore_step):
            fore_y[i, j] = y[i + his_step + j]
    print(hist_x.shape)
    print(fore_y.shape)
    return hist_x, fore_y


def TrainValidTest(x, y, train=0.8, valid=None):
    train_len = int(x.shape[0] * train)
    if valid is None:
        train_x = torch.tensor(x[:train_len], dtype=torch.float32)
        train_y = torch.tensor(y[:train_len], dtype=torch.float32)
        test_x = torch.tensor(x[train_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len:], dtype=torch.float32)
        return train_x, train_y, test_x, test_y
    else:
        valid_len = int(x.shape[0] * valid)
        train_x = torch.tensor(x[:train_len], dtype=torch.float32)
        train_y = torch.tensor(y[:train_len], dtype=torch.float32)
        valid_x = torch.tensor(x[train_len:train_len + valid_len], dtype=torch.float32)
        valid_y = torch.tensor(y[train_len:train_len + valid_len], dtype=torch.float32)
        test_x = torch.tensor(x[train_len + valid_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len + valid_len:], dtype=torch.float32)
        return train_x, train_y, valid_x, valid_y, test_x, test_y


def SaveNpy(data, dataname, filepath="..\\MeteoData\\"):
    data = np.array(data)
    np.save(filepath + dataname + ".npy", data)
    return


def Forecast(his_step, fore_step, x, y, lack=None):
    # x [num, city, feature]
    # y [num, city]
    RSME = []
    MAE = []
    SKILL = []
    feature_num = x.shape[2]
    num_hidden = 48
    num_layers = 1
    encoder = model_fs.EncoderGRU(feature_num, num_hidden, num_layers, device=device)
    decoder = model_fs.AttnDecoderRNN(feature_num, num_hidden, output_size=1)
    if cuda_gpu:
        encoder = encoder.to(device)
        decoder = decoder.to(device)
    enandde = model_fs.EncoderDecoderAtt(encoder=encoder, decoder=decoder, time_step=fore_step)
    criterion = nn.MSELoss()
    if cuda_gpu:
        enandde = enandde.to(device)
        criterion = criterion.to(device)
    optimizer = torch.optim.Adam(enandde.parameters(), lr=1e-2)
    for city in range(x.shape[1]):
        x_ones, _, _ = measure.Norm(x[:, city, :], 2)
        y_ones, y_min, y_max = measure.Norm(y[:, city], 1)
        hist_x, fore_y = Divide(his_step, fore_step, x_ones, y_ones)
        batch_size = 100
        epoch = 100
        train_x, train_y, test_x, test_y = TrainValidTest(hist_x, fore_y)
        train_loader = DataLoader(dataset=TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
                                  num_workers=4)
        for e in range(epoch):
            for i, data in enumerate(train_loader):
                inputs, target = data
                if cuda_gpu:
                    inputs, target = Variable(inputs).to(device), Variable(target).to(device)
                else:
                    inputs, target = Variable(inputs), Variable(target)
                # print("epoch:", e, i, "inputs:", inputs.shape, "target:", target.shape)
                outputs, attn_weights = enandde(inputs)
                loss = criterion(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if e % epoch == 0 or e + 1 == epoch:
                print('Epoch [{}/{}], Loss:{:.4f}'.format(e + 1, epoch, loss.item()))
        with torch.no_grad():
            test_x = test_x.to(device)
            test_outputs, _ = enandde(test_x)
            test_outputs = test_outputs.cpu()
            test_y = test_y * (y_max - y_min) + y_min
            test_outputs = test_outputs * (y_max - y_min) + y_min
            rmse = measure.GetRMSE(test_outputs, test_y)
            mae = measure.GetMAP(test_outputs, test_y)
            skill = measure.Skill(test_outputs, test_y)
            RSME.append(rmse)
            MAE.append(mae)
            SKILL.append(skill)

    SaveNpy(RSME, "RMSE" + str(lack))
    SaveNpy(MAE, "MAE" + str(lack))
    SaveNpy(SKILL, "SKILL" + str(lack))


def Mydataset(feature_num=[7, 8, 9, 12, 13, 14, 15, 17], targetnum=3):
    attribute_name = np.load("..\\MeteoData\\attribute_name.npy")
    dataset = np.load("..\\MeteoData\\KnowAir.npy")
    dataset = torch.tensor(dataset)
    print(dataset[0, 1, :])
    for i in range(len(feature_num)):
        if i == 0:
            newdataset = dataset[:, :, feature_num[i]].view(dataset.shape[0], dataset.shape[1], 1)
        else:
            newdataset = torch.cat(
                (newdataset, dataset[:, :, feature_num[i]].view(dataset.shape[0], dataset.shape[1], 1)), dim=2)
    newdataset = torch.cat((newdataset, dataset[:, :, targetnum].view(dataset.shape[0], dataset.shape[1], 1)), dim=2)
    newdataset = newdataset.reshape((newdataset.shape[0], 184, len(feature_num) + 1))
    np.save("..\\data\\Mydata.npy", newdataset)
    # specific_humidity，pm2.5，vwind + 950，total_precipitation，surface_pressure，
    # uwind + 950，vertical_velocity + 950，relative_humidity + 975
    # target 2m_temperature 3
    # 7, 8, 9, 12, 13, 14, 15, 17
    attribute_name = list(attribute_name)
    attribute_name.append('pm2.5')
    new_attribute = []
    # 回归目标放在最后
    for f_num in feature_num:
        new_attribute.append(attribute_name[f_num])
    new_attribute.append(attribute_name[targetnum])
    new_attribute = np.array(new_attribute)
    np.save("..\\data\\MyAttribute.npy", new_attribute)
    return


if __name__ == "__main__":
    data = np.load("../MeteoData/KnowAir.npy")
    # print(data.shape)
    # print(attribute_name.shape)
    # attribute = np.append(attribute_name[])
    x = np.concatenate((data[:, :, 0:3], data[:, :, 4:10], data[:, :, 12:]
                        ), axis=2)
    y = data[:, :, 3]
    # print(x.shape)
    # print(x.shape[1])
    # Mydataset()
    # label = np.load("..\\MeteoData\\MyAttribute.npy")
    # print(label)
    # data = np.load("..\\MeteoData\\MyData.npy")
    # print(data.shape)
    # ImportantInRF(x, y)
    '''
    print(x.shape[2])
    Forecast(his_step=16, fore_step=8, x=x, y=y)
    for i in range(x.shape[2]):
        x = np.concatenate((x[:, :, :i], x[:, :, i+1:]), axis=2)
        Forecast(his_step=16, fore_step=8, x=x, y=y, lack=i)
    #Divide(his_step=16, fore_step=8, x=x, y=y)

    '''

