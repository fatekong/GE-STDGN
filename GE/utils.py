#!/usr/bin/env python
# -*- coding:utf-8 -*-

import functools
import random
from GE.population import Population
from GE.individual import Individual
#from Graph.Main import AllFuntion
import numpy as np
import torch
from parallel import MultiPro
import time
import sys
sys.path.append('..')
from tester import fitness_cal
# import Graph.Main as GM



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NSGA2Utils(object):

    def __init__(self, pop=30, iter=30, cross_pro=0.7, mutation_pro=0.05, matrix_size=184, tourna_size=5, index=None, index_group=None):
        self.pop_num = pop
        self.iter_num = iter
        self.cross_pro = cross_pro
        self.mutation_pro = mutation_pro
        self.matrix_size = matrix_size
        self.max_index = None
        self.index_group = index_group
        if index is not None:
            self.lens = len(index)
        if index_group is not None:
            self.station_pro = np.load('../data/station_pro.npy')
            self.lens = len(self.station_pro)
            print(self.lens)
        else:
            self.lens = int(183 * 184 / 2)
        #锦标赛选择大小
        self.tourna_size = tourna_size
        self.Fitness = []
        self.hist_fit = []
        # cuda_num: number of CUDAs are used in graph  evolution
        self.parallel = MultiPro(cuda_num=3, fun=self.calculate_objectives)

    def fast_nondominated_sort(self, population):
        population.fronts = []
        population.fronts.append([])
        i = 0
        for individual in population:
            individual.domination_count = 0
            #individual.dominated_solutions = set()
            individual.dominated_solutions = []
            for other_individual in population:
                i += 1
                if individual == other_individual:
                    continue
                if individual.dominates(other_individual):
                    # set
                    #individual.dominated_solutions.add(other_individual)
                    # []
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                population.fronts[0].append(individual)
                individual.rank = 0
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    def __sort_objective(self, val1, val2, m):
        if val1.objectives[m] < val2.objectives[m]:
            return -1
        elif val1.objectives[m] == val2.objectives[m]:
            return 0
        else:
            return 1

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front = sorted(front, key=functools.cmp_to_key(functools.partial(self.__sort_objective, m=m)))
                #开始和结尾都设为正无穷
                front[0].crowding_distance = float('inf')
                front[solutions_num - 1].crowding_distance = float('inf')
                max_objective = front[solutions_num - 1].objectives[m]
                min_objective = front[0].objectives[m]
                # avoid value 0 by max and min
                if max_objective == min_objective:
                    max_objective += 0.01
                #print('max ob: ', max_objective)
                #print('min ob: ', min_objective)
                for index, value in enumerate(front[1:solutions_num - 1]):
                    if max_objective - min_objective != float(0):
                        front[index].crowding_distance = float('inf')
                    else:
                        front[index].crowding_distance = (front[index + 1].crowding_distance -
                                                          front[index - 1].crowding_distance) / (max_objective - min_objective)

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_initial_population(self, other_adj=None):
        pop = Population()
        if self.index_group is not None:
            start = self.index_group[0]
            end = self.index_group[1]
            self.pop_num = end - start + 1
            max_edge = np.load('../data/index_' + str(end) + '.npy')
            self.max_index = max_edge
            #self.edge_index = max_edge
            while start <= end:
                individual = Individual()
                thisindex = np.load('../data/index_' + str(start) + '.npy')
                initials = np.zeros(self.lens)
                j = 0
                for i in range(self.lens):
                    if (self.max_index[i] == thisindex[j]).all():
                        initials[i] = 1
                        j += 1
                    if j == thisindex.shape[0]:
                        break
                start += 1
                individual.compression = initials
                individual.index = self.max_index
                individual.matrix_size = self.matrix_size
                #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                #individual = self.calculate_objectives(individual, device)
                pop.population.append(individual)
            if other_adj is not None:
                for adj in other_adj:
                    individual = Individual()
                    individual.specialty = True
                    individual.index = self.max_index
                    individual.matrix_size = self.matrix_size
                    individual.adj = adj
                    individual.matrix_to_index()
                    pop.population.append(individual)
            start = time.time()
            self.parallel.P(population=pop.population)
            #print('---------------------------------------------------------')
            pop.population = self.parallel.R
            end = time.time()
            print('time cost: ', end-start)
            #print('R num:', len(self.parallel.R))
            print(self.parallel.index)
            print('initialization over!')
            return pop

    def calculate_objectives(self, individual, device):
        adj_i = individual.index_to_matrix()
        #torch
        #adj_i = torch.tensor(adj_i + np.eye(self.matrix_size), dtype=torch.float32)
        #np
        adj_i = adj_i + np.eye(self.matrix_size)
        sparse = individual.get_sparse()
        #适应度函数，主要计算矩阵精确度
        fit, model = fitness_cal(adjmat=adj_i, device=device)
        #fit, model, device = OtherFitness(adj_i, device)
        #print(fit, sparse)
        individual.model_parameter = model.state_dict()
        individual.objectives = [fit, sparse]
        return individual

    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            #print('select parent')
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            #print('cross over!')
            child1 = self.__mutate(child1)
            child2 = self.__mutate(child2)
            #print('mutation over!')
            #print('child1: ')
            #------------------------------------not parallel----------------------------------
            #child1 = self.calculate_objectives(child1)
            #print('child2: ')
            #child2 = self.calculate_objectives(child2)
            # ------------------------------------not parallel----------------------------------
            #print('calculation over!')
            children.append(child1)
            children.append(child2)
        print('creat over!')
        # ------------------------------------parallel--------------------------------------
        start = time.time()
        self.parallel.P(children)
        end = time.time()
        # ------------------------------------parallel--------------------------------------
        print('time cost: ', end - start)
        #print(len(self.parallel.R))
        print(self.parallel.index)
        children = self.parallel.R
        return children

    def __crossover(self, individual1, individual2):
        rand = np.random.random()

        # accumulation: Prevent cross update selection from entering dead loop
        accumulation = 0
        if rand < self.cross_pro:
            while True:
                cross_start = np.random.randint(0, self.lens - 1)
                child1 = Individual()
                child2 = Individual()
                child1.compression = individual1.compression
                child1.matrix_size = individual1.matrix_size
                child1.index = individual1.index
                child2.compression = individual2.compression
                child2.matrix_size = individual2.matrix_size
                child2.index = individual2.index
                child1.objectives = None
                child2.objectives = None
                child1.compression[cross_start:] = individual2.compression[cross_start:]
                child2.compression[cross_start:] = individual1.compression[cross_start:]
                diff1 = np.sum(np.abs(child1.compression - individual1.compression))
                diff2 = np.sum(np.abs(child1.compression - individual2.compression))
                if diff1 + diff2 < 100 and accumulation != 5:
                    del child1
                    del child2
                    accumulation += 1
                    continue
                else:
                    break
        else:
            child1 = Individual()
            child2 = Individual()
            child1.compression = individual1.compression
            child1.matrix_size = individual1.matrix_size
            child1.index = individual1.index
            child2.compression = individual2.compression
            child2.matrix_size = individual2.matrix_size
            child2.index = individual2.index
            child1.compression = individual1.compression
            child2.compression = individual2.compression
            child1.objectives = None
            child2.objectives = None
            #print('equal: ', child1 == child2)
        return child1, child2

    #@profile
    def __mutate(self, child):
        rand = np.random.random()
        if rand < self.mutation_pro:
            index = np.random.choice(self.lens, int(self.mutation_pro * self.lens), replace=False)
            # Mutation based on distance
            # rand_trans = np.random.random()
            for p in index:
                child.compression[p] = 1 - child.compression[p]
                # if child.compression[p] == 1:
                #     if rand_trans < self.station_pro[p]:
                #         child.compression[p] = 1 - child.compression[p]
                # else:
                #     if rand_trans > self.station_pro[p]:
                #         child.compression[p] = 1 - child.compression[p]
        return child

    def __tournament(self, population):
        participants = random.sample(population, self.tourna_size)
        best = None
        for participant in participants:
            if best is None or self.crowding_operator(participant, best) == 1:
                best = participant
        return best
