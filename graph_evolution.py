from GE.utils import NSGA2Utils
from GE.population import Population
import numpy as np
import torch
import functools

class Evolution(object):
    # 如果有精英保留机制，就直接去重
    def __init__(self, pop=50, iter=30, cross_pro=0.8, mutation_pro=0.2, matrix_size=184, tourna_size=5, index=None,
                 index_group=None):
        self.utils = NSGA2Utils(pop=pop, iter=iter, cross_pro=cross_pro, mutation_pro=mutation_pro,
                                matrix_size=matrix_size,
                                tourna_size=tourna_size, index=index, index_group=index_group)

        self.population = None
        self.num_of_generations = iter
        self.on_generation_finished = []
        self.history = []

    def register_on_new_generation(self, fun):
        self.on_generation_finished.append(fun)

    def evolve(self, other_adj=None):
        self.population = self.utils.create_initial_population(other_adj=other_adj)
        self.num_of_individuals = len(self.population)
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        # children = self.utils.create_initial_population()
        children = self.utils.create_children(self.population.population)
        print(len(children))
        returned_population = None
        for i in range(self.num_of_generations):
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            new_population = Population()
            front_num = 0
            print(self.num_of_individuals)
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            sorted(self.population.fronts[front_num], key=functools.cmp_to_key(self.utils.crowding_operator))
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
            returned_population = self.population
            self.population = new_population
            print('children:')
            children = self.utils.create_children(self.population.population)
            print('-----------------------------------------over-------------------------------------------')
            for fun in self.on_generation_finished:
                value, his = fun(returned_population, i)
                if his == 'his':
                    self.history.append(value)
        return list(set(returned_population.fronts[0]))

    def acc_best(self, fronts):
        best_value = np.zeros(len(fronts[0].objectives))
        best_value[:] = float('inf')
        best_individual = []
        for m in range(len(fronts[0].objectives)):
            best_individual.append(None)
            for f in fronts:
                if best_value[m] > f.objectives[m]:
                    best_value[m] = f.objectives[m]
                    best_individual[m] = f

        return best_individual


def print_genetation(population, genetation):
    G_set = {}
    G_obj = list(set(population.fronts[0]))
    G_set[genetation] = []
    for p in G_obj:
        G_set[genetation].append(p.objectives)
    return G_set, 'his'

# if __name__ == "__main__":
#     TIN = np.load('../data/TIN_adj.npy') - np.eye(184)
#     Dis = np.load('../data/Geo_adj.npy') - np.eye(184)
#     other_adj = [TIN, Dis]
#     E = Evolution(pop=20, iter=20, cross_pro=0.8, mutation_pro=0.1, index_group=[6, 23])
#     E.register_on_new_generation(print_genetation)
#     adj_set = []
#     front = E.evolve(other_adj=other_adj)
#     np.save('data/dictionary.npy', E.history)
#     print(len(front))
#     last_front = []
#     for f in front:
#         print(f.objectives)
#         last_front.append(f.adj)
#     np.save("data/last_front.npy", last_front)
#     best = E.acc_best(front)
#     print('--------------------------------best--------------------------------')
#     model_par = 0
#     for b in best:
#         adj = b.index_to_matrix() + np.eye(b.matrix_size)
#         torch.save(b.model_parameter, 'data/model_par' + str(model_par))
#         model_par += 1
#         print(adj)
#         adj_set.append(adj)
#         print(b.objectives)
#     np.save('data/AdjSet(GE).npy', adj_set)
