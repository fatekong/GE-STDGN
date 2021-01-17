#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham

city_fp = '../data/city.txt'
altitude_fp = '../data/altitude.npy'

class Graph():
    def __init__(self, dist_thres=3, alti_thres=1200, weight=0.8, use_altitude=True):
        # the threshold of distance in geo
        self.dist_thres = dist_thres
        # factor to weight altitude and distance
        self.weight = weight

        # the threshold of distance is 1200 in geo
        self.alti_thres = alti_thres
        self.use_altitude = use_altitude

        self.altitude = self._load_altitude()
        # information of nodes
        self.nodes = self._gen_nodes()
        # altitude between nodes
        self.node_attr = self._add_node_attr()
        self.node_num = len(self.nodes)

        self.edge_index, self.edge_attr = self._gen_edges()
        if self.use_altitude:
            self._update_edges()
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]

    def _load_altitude(self):
        assert os.path.isfile(altitude_fp)
        altitude = np.load(altitude_fp)
        print('altitude.shape', altitude.shape)
        return altitude

    def _lonlat2xy(self, lon, lat, is_aliti):
        if is_aliti:
            lon_l = 100.0
            lat_u = 48.0
            res = 0.05
        else:
            lon_l = 103.0
            lat_u = 42.0
            res = 0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def _gen_nodes(self):
        nodes = OrderedDict()
        with open(city_fp, 'r') as f:
            for line in f:
                idx, city, lon, lat = line.rstrip('\n').split(' ')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                x, y = self._lonlat2xy(lon, lat, True)
                altitude = self.altitude[y, x]
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):
        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))
        return lines

    def _gen_pro(self, index):
        # Calculating Euclidean distance
        dist_mat = np.zeros((self.node_num, self.node_num))
        heit_mat = np.zeros((self.node_num, self.node_num))
        i = 0
        while i < self.node_num:
            j = i+1
            while j < self.node_num:
                src_lat, src_lon = self.nodes[i]['lat'], self.nodes[i]['lon']
                dest_lat, dest_lon = self.nodes[j]['lat'], self.nodes[j]['lon']
                src_location = (src_lat, src_lon)
                dest_location = (dest_lat, dest_lon)
                dist_km = geodesic(src_location, dest_location).kilometers
                src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
                dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
                points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1, 0))
                altitude_points = self.altitude[points[0], points[1]]
                altitude_src = self.altitude[src_y, src_x]
                altitude_dest = self.altitude[dest_y, dest_x]
                max_src = np.max(altitude_points - altitude_src)
                max_dest = np.max(altitude_points - altitude_dest)
                # The altitude of the mountains
                if max_src > max_dest:
                    max_altitude = max_src
                else:
                    max_altitude = max_dest
                dist_mat[i][j] = dist_mat[j][i] = dist_km
                heit_mat[i][j] = heit_mat[j][i] = max_altitude
                j += 1
            i += 1
        max_dis = np.max(dist_mat, axis=1).reshape(self.node_num, 1)
        min_dis = np.min(dist_mat, axis=1).reshape(self.node_num, 1)
        max_h = np.max(heit_mat, axis=1).reshape(self.node_num, 1)
        min_h = np.min(heit_mat, axis=1).reshape(self.node_num, 1)
        dist_mat = (dist_mat - min_dis)/(max_dis - min_dis)
        heit_mat = (heit_mat - min_h)/(max_h - min_h)
        total = self.weight * dist_mat + (1 - self.weight) * heit_mat
        edge_index = []
        edge_value = []
        for i in range(self.node_num):
            temp = []
            total[i, i] = float('inf')
            total_line = list(total[i, :])
            inf = np.zeros(total[i, :].shape) + float('inf')
            for j in range(index):
                temp.append(total_line.index(min(total_line)))
                total_line[total_line.index(min(total_line))] = float('inf')
                if (np.array(total_line) == inf).all():
                    break
            temp.sort()
            for t in temp:
                edge_index.append([i, t])
                edge_value.append(total[i, t])
        return np.array(edge_index), np.array(edge_value)

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        #计算欧几里得距离
        dist = distance.cdist(coords, coords, 'euclidean')
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        #得到小于距离阈值的adj
        adj[dist <= self.dist_thres] = 1
        #print(adj)
        assert adj.shape == dist.shape
        dist = dist * adj
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        edge_index, dist = edge_index.numpy(), dist.numpy()
        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            #src和dest分别是两个顶点的
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            #两个点的经纬度距离
            v, u = src_lat - dest_lat, src_lon - dest_lon
            #经纬度的风向，u，v风
            u = u * units.meter / units.second
            v = v * units.meter / units.second

            direc = mpcalc.wind_direction(u, v)._magnitude
            #风速情况列表
            direc_arr.append(direc)
            #地理距离列表
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        #地理距离和风传距离
        attr = np.stack([dist_arr, direc_arr], axis=-1)
        return edge_index, attr

    #对高度进行进一步选择
    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:,i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)

    def RecoverMatrix(self):
        adjmatrix = np.zeros((self.node_num, self.node_num), dtype=float)
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            adjmatrix[src][dest] = self.edge_attr[i][0]
        return adjmatrix

if __name__ == "__main__":
    graph = Graph()
    adj = graph.RecoverMatrix()
    adj[adj > 1] = 1
    Geo_threshold = adj + np.eye(graph.node_num)
    print(Geo_threshold)