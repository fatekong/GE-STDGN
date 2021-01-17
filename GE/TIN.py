#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import utils
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import torch
class TIN_Point:
    X = 0
    Y = 0
    feature = 0
    Num = 0
    Name = ''
    def __init__(self, x, y, feature=None, n=0, name = ''):
        self.X = x
        self.Y = y
        self.Num = n
        self.feature = feature
        self.Name = name

    def Infor(self):
        return "NUM " + str(self.Num) + "X: " + str(self.X) + ", Y: " + str(self.Y)

    def Equals(self, obj):
        if type(obj) is TIN_Point:
            if obj.X == self.X and obj.Y == self.Y:
                return True
        return False

class TIN_Edge:
    SignT1 = -1
    SignT2 = -1
    def __init__(self, mp1, mp2):
        self.p1 = mp1
        self.p2 = mp2

    def SetTriangleSign(self, sign):
        if self.SignT1 == -1:
            self.SignT1 = sign
        else:
            self.SignT2 = sign

    def Equals(self, obj):
        if type(obj) is TIN_Edge:
            if (obj.p1.Equals(self.p1) and obj.p2.Equals(self.p2)) \
                or (obj.p2.Equals(self.p1) and obj.p1.Equals(self.p2)):
                return True
        return False

    def Infor(self):
        return self.p1.Infor() + "\n" + self.p2.Infor()

class TIN_Circle:
    X = 0
    Y = 0
    R_pow = 0
    def __init__(self, mx=0, my=0, mr_pow=0):
        self.X = mx
        self.Y = my
        self.R_pow = mr_pow

    def Infor(self):
        return "center of the circle：" + str(self.X) + "," + str(self.Y) + "--" + str(self.R_pow)

class TIN_Triangle:
    outcircle = 0
    def __init__(self, mp1, mp2, mp3, circle=None):
        self.p1 = mp1
        self.p2 = mp2
        self.p3 = mp3
        self.outcircle = circle

    def Equals(self, obj):
        if type(obj) is TIN_Triangle:
            if obj.p1.Equals(self.p1) or obj.p1.Equals(self.p2) or obj.p1.Equals(self.p3):
                if obj.p2.Equals(self.p1) or obj.p2.Equals(self.p2) or obj.p2.Equals(self.p3):
                    if obj.p3.Equals(self.p1) or obj.p3.Equals(self.p2) or obj.p3.Equals(self.p3):
                        return True
        return False

    def Infor(self):
        s = ""
        s += "p1: (" + str(self.p1.X) + "," + str(self.p1.Y) + ") :num&value: " + str(self.p1.Num) + "\n"
        s += "p2: (" + str(self.p2.X) + "," + str(self.p2.Y) + ") :num&value: " + str(self.p2.Num) + "\n"
        s += "p3: (" + str(self.p3.X) + "," + str(self.p3.Y) + ") :num&value: " + str(self.p3.Num) + "\n"
        return s

class Delaunay:
    EPSILON = 1.0 / 1048576.0
    def __init__(self):
        return

    def SuperTriangle(self, list):
        xmax = 0
        xmin = 0
        ymax = 0
        ymin = 0
        for i in range(len(list)):
            if i == 0:
                xmax = xmin = list[i].X
                ymax = ymin = list[i].Y
            else:
                if list[i].X < xmin:
                    xmin = list[i].X
                if list[i].X > xmax:
                    xmax = list[i].X
                if list[i].Y < ymin:
                    ymin = list[i].Y
                if list[i].Y > ymax:
                    ymax = list[i].Y
        dx = xmax - xmin
        dy = ymax - ymin
        dmax = max(dx, dy)
        xmid = xmin + dx / 2
        ymid = ymin + dy / 2
        super = TIN_Triangle(TIN_Point(xmid - 5 * dmax, ymid -dmax, 0, -1),
                             TIN_Point(xmid, ymid + 5 * dmax, 0, -2),
                             TIN_Point(xmid + 5 * dmax, ymid - dmax, 0, -3))
        mycircle = self.Out_Circle(super)
        super.outcircle = mycircle
        return super

    def Out_Circle(self, t):
        px1 = t.p1.X
        py1 = t.p1.Y
        px2 = t.p2.X
        py2 = t.p2.Y
        px3 = t.p3.X
        py3 = t.p3.Y
        fabsy1y2 = abs(py1 - py2)
        fabsy2y3 = abs(py2 - py3)
        if fabsy1y2 < self.EPSILON and fabsy2y3 < self.EPSILON:
            circle = TIN_Circle()
            return circle

        if fabsy1y2 < self.EPSILON:
            m2 = -((px3 - px2) / (py3 - py2))
            mx2 = (px2 + px3) / 2.0
            my2 = (py2 + py3) / 2.0
            xc = (px2 + px1) / 2.0
            yc = m2 * (xc - mx2) + my2

        elif fabsy2y3 < self.EPSILON:
            m1 = -((px2 - px1) / (py2 - py1))
            mx1 = (px1 + px2) / 2.0
            my1 = (py1 + py2) / 2.0
            xc = (px3 + px2) / 2.0
            yc = m1 * (xc - mx1) + my1

        else:
            m1 = -((px2 - px1) / (py2 - py1))
            m2 = -((px3 - px2) / (py3 - py2))
            mx1 = (px1 + px2) / 2.0
            mx2 = (px2 + px3) / 2.0
            my1 = (py1 + py2) / 2.0
            my2 = (py2 + py3) / 2.0
            if m1 - m2 == 0:
                #说明三边在一条线上
                return None
            xc = (m1 * mx1 - m2 * mx2 + my2 - my1) / (m1 - m2)
            yc = m1 * (xc - mx1) + my1 if fabsy1y2 > fabsy2y3 else m2 * (xc - mx2) + my2

        dx = px2 - xc
        dy = py2 - yc
        return TIN_Circle(xc, yc, dx * dx + dy * dy)

    def Updata(self, edges):
        flag = False
        i = 0
        while i < len(edges):
            j = i+1
            while j < len(edges):
                if edges[i].Equals(edges[j]):
                    edges.pop(j)
                    flag = True
                    j -= 1
                j += 1
            if flag:
                edges.pop(i)
                i -= 1
                flag = False
            i += 1
        return edges

    def RomoveSuper(self, super, other):
        if super.p1.Equals(other.p1) or \
           super.p2.Equals(other.p1) or \
           super.p3.Equals(other.p1):
            return True
        if super.p1.Equals(other.p2) or \
           super.p2.Equals(other.p2) or \
           super.p3.Equals(other.p2):
            return True
        if super.p1.Equals(other.p3) or \
           super.p2.Equals(other.p3) or \
           super.p3.Equals(other.p3):
            return True
        return False

    def ConstructionDelaunay(self, vertices):
        num = len(vertices)
        if num < 3:
            return None
        vertices.sort(key=lambda x:x.X, reverse=False)
        for i in range(len(vertices)):
            print(vertices[i].Infor())
        print('sort completed')
        super = self.SuperTriangle(vertices)
        print(super.Infor())
        print('super completed')
        # 外三角
        open = []
        # 内三角
        closed = []
        # 边
        edges = []
        open.append(super)
        print('adjoined super')
        for i in range(num):
            edges = []
            print("i: %d, num: %d" %(i, num))
            thepoint = vertices[i]
            j = 0
            while j < len(open):
                print("j: %d, open.num: %d" %(j, len(open)))
                print(open[j].outcircle.Infor())
                dx = thepoint.X - open[j].outcircle.X
                if dx > 0 and dx ** 2 > open[j].outcircle.R_pow:
                    print("Point %d is on the right side of the circle. This triangle is Delaunay triangle. Add this triangle from open to close!"%thepoint.Num)
                    closed.append(open[j])
                    open.pop(j)
                    continue
                dy = thepoint.Y - open[j].outcircle.Y
                if dx ** 2 + dy ** 2 - open[j].outcircle.R_pow > self.EPSILON:
                    print("Point %d is outside the circle and not on the right side. The triangle is not sure to be Delaunay triangle. Do not do any operation!"%thepoint.Num)
                    j += 1
                    continue
                print("Point %d is on the inside of the circle，The triangle must not be a Delaunay triangle. Remove the triangle and add three sides to the sideset." %thepoint.Num)
                edges.append(TIN_Edge(open[j].p1, open[j].p2))
                edges.append(TIN_Edge(open[j].p1, open[j].p3))
                edges.append(TIN_Edge(open[j].p2, open[j].p3))
                open.pop(j)
            self.Updata(edges)
            for j in range(len(edges)):
                print(edges[j].Infor())
                newtriangle = TIN_Triangle(edges[j].p1, edges[j].p2, thepoint)
                newtriangle.outcircle = self.Out_Circle(newtriangle)
                if newtriangle.outcircle is None:
                    continue
                open.append(newtriangle)
            #ShowDelaunay(open)
        print('completed')
        for i in range(len(open)):
            closed.append(open[i])
        open = []
        for i in range(len(closed)):
            if self.RomoveSuper(closed[i], super) is False:
                open.append(closed[i])
        return open

    def GetAdjacencyMat(self, triangles, num):
        admat = np.zeros((num, num))
        for i in range(len(triangles)):
            admat[triangles[i].p1.Num][triangles[i].p2.Num] = 1
            admat[triangles[i].p1.Num][triangles[i].p3.Num] = 1
            admat[triangles[i].p2.Num][triangles[i].p3.Num] = 1
            admat[triangles[i].p2.Num][triangles[i].p1.Num] = 1
            admat[triangles[i].p3.Num][triangles[i].p1.Num] = 1
            admat[triangles[i].p3.Num][triangles[i].p2.Num] = 1
        return admat

# this function is adopted to draw the TIN using matplotlib
# open it the triangle formed to Circumscribed circle
def ShowDelaunay(open=None, vertices=None, adjmat=None):
    if vertices is not None:
        x = []
        y = []
        for point in vertices:
            x.append(point.X)
            y.append(point.Y)
        plt.title("Delaunay")
        plt.xlabel("lon")
        plt.ylabel("lat")
        plt.scatter(x, y, s=20, c='b', marker='o')
    if open is not None:
        for tri in open:
            x = []
            y = []
            x.append([tri.p1.X, tri.p2.X])
            x.append([tri.p2.X, tri.p3.X])
            x.append([tri.p3.X, tri.p1.X])
            y.append([tri.p1.Y, tri.p2.Y])
            y.append([tri.p2.Y, tri.p3.Y])
            y.append([tri.p3.Y, tri.p1.Y])
            plt.plot(x, y)
    plt.legend()
    plt.show()

def ShowCircle(x, y, radius):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cir = Circle(xy=(x, y), radius=radius, alpha=0.1)
    ax.add_patch(cir)
    plt.plot(x,y,'ro')
    plt.show()

def Readcity(filepath):
    pos = []
    with open(filepath, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            p_tmp = [i for i in lines.split()]
            pos.append(p_tmp)
            pass
    return pos

def GetDelaunayAdjMat(filepath=None):
    if filepath is None:
        print("there is no input filepath.")
        return None
    cities = Readcity(filepath)
    vertices = []
    for city in cities:
        vertices.append(TIN_Point(x=float(city[2]), y=float(city[3]), n=int(city[0]), name=city[1]))
    mydelaunay = Delaunay()
    open = mydelaunay.ConstructionDelaunay(vertices)
    adjmat = mydelaunay.GetAdjacencyMat(open, len(vertices))
    adjmat += np.eye(adjmat.shape[0])
    return adjmat

if __name__ == "__main__":
    filepath = '../data/city.txt'
    adjmat = GetDelaunayAdjMat(filepath)
    np.save("../data/adjmat", adjmat)

