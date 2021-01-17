#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import measure

from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda_gpu = torch.cuda.is_available()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

def TrainValidTest(x, y, train=0.8, valid=None, shuffle=True):

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
        _x = x[train_len:]
        _y = y[train_len:]

        if shuffle:
            setup_seed(20)
            np.random.shuffle(_x)
            setup_seed(20)
            np.random.shuffle(_y)
        valid_x = torch.tensor(_x[:train_len+valid_len], dtype=torch.float32)
        valid_y = torch.tensor(_y[:train_len+valid_len], dtype=torch.float32)
        test_x = torch.tensor(x[train_len+valid_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len+valid_len:], dtype=torch.float32)
        return train_x, train_y, valid_x, valid_y, test_x, test_y

def Divide(x, y, his_step, pre_step):
    x_shape = len(x.shape)
    if x_shape == 3:
        hist_x = np.zeros((x.shape[0] - his_step - pre_step + 1, x.shape[1], his_step, x.shape[2]))
    else:
        hist_x = np.zeros((x.shape[0] - his_step - pre_step + 1, x.shape[1], his_step))
    fore_y = np.zeros((y.shape[0] - his_step - pre_step + 1, y.shape[1], pre_step))
    for i in range(x.shape[0] - his_step - pre_step + 1):
        for j in range(his_step):
            if x_shape == 3:
                hist_x[i, :, j, :] = x[i+j, :, :]
            else:
                hist_x[i, :, j] = x[i + j, :]
        for j in range(pre_step):
            fore_y[i, :, j] = y[i+his_step+j, :]
    print(hist_x.shape)
    print(fore_y.shape)
    return hist_x, fore_y


class trainer():
    def __init__(self, model, epoch, batch, lr, decay, his_step, pre_step, x, y, train=0.8, valid=None, device=None):
        self.epoch = epoch
        self.batch = batch
        self.model = model
        self.lr = lr
        self.decay = decay
        self.x = x
        self.y = y
        self.his = his_step
        self.pre = pre_step
        self.valid = valid
        self.trainset = train

    def train(self):
        print("device: ", self.device)
        train_loss = []
        valid_rsme = []

        nodes = self.x.shape[1]
        feature_num = self.x.shape[-1]
        criterion = nn.MSELoss()
        if cuda_gpu:
            model = self.model.to(device)
            criterion = criterion.to(device)

        model = model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.decay)

        x_ones, x_min, x_max = measure.Norm(np.array(torch.tensor(self.x).view(-1, feature_num), dtype=float), dim=2)
        y_ones, y_min, y_max = measure.Norm(np.array(torch.tensor(self.y).view(-1), dtype=float), dim=1)

        source_x = torch.tensor(x_ones).view(-1, nodes, feature_num)
        source_y = torch.tensor(y_ones).view(-1, nodes)

        hist_x, fore_y = Divide(source_x, source_y, self.his, self.pre)
        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, train=self.trainset, valid=0.05)

        setup_seed(20)
        train_loader = DataLoader(dataset=TensorDataset(train_x, train_y), batch_size=self.batch, shuffle=True)

        Valid_y = valid_y * (y_max - y_min) + y_min
        for e in range(self.epoch):
            for i, data in enumerate(train_loader):
                inputs, target = data
                if cuda_gpu:
                    inputs, target = Variable(inputs).to(device), Variable(target).to(device)
                else:
                    inputs, target = Variable(inputs), Variable(target)

                y_hat = model(inputs)
                loss = criterion(y_hat, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (e + 1) % 10 == 0:
                # print(device)
                print('Epoch [{}/{}], Loss:{:.4f}'.format(e + 1, self.epoch, loss.item()))
                train_loss.append(loss.item())
                with torch.no_grad():
                    valid_x = valid_x.to(device)
                    MyModel = MyModel.eval()
                    valid_outputs = MyModel(valid_x)
                    valid_outputs = valid_outputs.cpu()
                    valid_outputs = valid_outputs * (y_max - y_min) + y_min
                    rmse = measure.GetRMSE(valid_outputs, Valid_y)
                    valid_rsme.append(rmse)
                    # mae = Measure.GetMAP(valid_outputs, Valid_y)
                    # skill = Measure.Skill(valid_outputs, Valid_y)
                    # valid_rmse.append(rmse)
                    print("valid RMSE:", rmse)
                    # print("valid MAE:", mae)
                    # print("valid SKILL:", skill)
                    model = model.train()

        np.save("valid_RSME" + model.name, np.array(valid_rsme))
        np.save("train_Loss" + model.name, np.array(train_loss))

        if self.valid is not None:
            print('valid result: ', rmse)
            return rmse, model.cpu()

        with torch.no_grad():
            model = model.eval()
            test_x = test_x.to(device)
            # print(test_y.shape)
            # test_outputs = T_GCN(test_x)
            test_outputs = MyModel(test_x)
            test_outputs = test_outputs.cpu()
            test_y = test_y * (y_max - y_min) + y_min
            test_outputs = test_outputs * (y_max - y_min) + y_min
            rmse = measure.GetRMSE(test_y, test_outputs)
            # mae = Measure.GetMAP(test_y, test_outputs)
            # mape = Measure.GetMAPE(test_y, test_outputs)
            # r2 = Measure.R_square(test_y, test_outputs)
            print("RMSE:", rmse)
            # print("MAE:", mae)
            # print("MAPE:", mape)
            # print("R2:", r2)
            # print(train_loss)
            # print(valid_rsme)
        return rmse, model.cpu()
