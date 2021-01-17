#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from Geo_threshold import Graph
import numpy as np
if __name__ == "__main__":
    graph = Graph()
    k = 6
    while k <= 23:
        print(k)
        edge_index, edge_value = graph._gen_pro(k)
        np.save("../data/index_" + str(k), edge_index)
        if k == 23:
            np.save("../data/station_pro.npy", edge_value)
        k += 1