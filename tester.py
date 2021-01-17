#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import time
from trainer import trainer, Divide, TrainValidTest
from model import STDGN_woa
import measure
import utils
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fitness_cal(adjmat, device=device):
    adjmat = torch.tensor(adjmat, dtype=torch.float32)
    start = time.time()
    dataset = np.load("../data/DataSet.npy")
    x = dataset[:5848, :, :]
    y = dataset[:5848, :, -1]
    nodes = x.shape[1]
    features = x.shape[-1]
    model = STDGN_woa(num_units=64, adj=adjmat, num_nodes=nodes, num_feature=features,
                  seq_len=16, pre_len=8, device=device, deep=1)

    model_train = trainer(model, epoch=200, batch=200, lr=1e-3, decay=0, his_step=16, pre_step=8, x=x, y=y, train=0.8, valid='yes', device=device)
    RMSE, model = model_train.train()

    end = time.time()
    print("time cost: ", end-start)
    return RMSE, model

def test(model, his_step, pre_step, x, y):
    print("device: ", device)
    feature_num = x.shape[-1]
    nodes = x.shape[1]
    x_ones, x_min, x_max = measure.Norm(np.array(torch.tensor(x).view(-1, feature_num), dtype=float), dim=2)
    y_ones, y_min, y_max = measure.Norm(np.array(torch.tensor(y).view(-1), dtype=float), dim=1)
    source_x = torch.tensor(x_ones).view(-1, nodes, feature_num)
    source_y = torch.tensor(y_ones).view(-1, nodes)
    hist_x, fore_y = Divide(source_x, source_y, his_step, pre_step)
    _, _, _, _, test_x, test_y = TrainValidTest(hist_x, fore_y, valid=0.05)
    model = model.to(device)
    with torch.no_grad():
        model = model.eval()
        test_x = test_x.to(device)
        test_outputs = model(test_x)
        test_outputs = test_outputs.cpu()
        test_y = test_y * (y_max - y_min) + y_min
        test_outputs = test_outputs * (y_max - y_min) + y_min
    rmse = measure.GetRMSE(test_outputs, test_y)
    mae = measure.GetMAP(test_outputs, test_y)
    skill = measure.Skill(test_outputs, test_y)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("SKILL:", skill)
    for i in range(pre_step):
        print('step ', i)
        to = test_outputs[:, :, i]
        ty = test_y[:, :, i]
        rmse = measure.GetRMSE(ty, to)
        mae = measure.GetMAP(ty, to)
        mape = measure.GetMAPE(ty, to)
        r2 = measure.R_square(ty, to)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("MAPE:", mape)
        print("R2:", r2)
