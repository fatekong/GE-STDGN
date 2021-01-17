#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from multiprocessing import Manager
from multiprocessing.pool import Pool
import multiprocessing as mp
import time
import torch

class MultiPro():
    def __init__(self, cuda_num=3, fun=None):
        self.index = []
        self.lists = []
        for i in range(cuda_num):
            cuda_name = 'cuda:' + str(i)
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            self.lists.append(device)
        self.pop = None
        self.R = []
        self.fun = fun
        self.set_start_method = True

    def Run(self, parameters, stack, i):
        while True:
            try:
                pop = stack.pop(0)
            except BaseException:
                time.sleep(1)
            else:
                break
        r = self.fun(parameters, pop)
        return r, pop, i

    def a(self, value):
        r, pop, i = value
        self.R.append(r)
        self.stack.append(pop)
        self.index.append(i)
        return

    def P(self, population=None):
        if self.set_start_method:
            mp.set_start_method('spawn')
        self.set_start_method = False
        self.index = []
        self.R = []
        self.pop = population
        manager = Manager()
        self.stack = manager.list()
        self.stack.extend(self.lists)
        pool = Pool(processes=len(self.stack))
        # The CUDA device number is pushed into the stack
        for i in range(len(self.pop)):
            pool.apply_async(self.Run, args=(self.pop[i], self.stack, i), callback=self.a)
        pool.close()
        pool.join()
        print('over')
        return

    def Judge(self):
        end = time.time()
        self.P()
        ended = time.time()
        print(ended - end)

