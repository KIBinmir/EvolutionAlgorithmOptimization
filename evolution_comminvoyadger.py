from random import randint

import pygame as pg
import numpy as np

def dist(p1, p2):
    return np.linalg.norm(np.subtract(p1, p2))

def getRouteLen(p0, pts, inds):
    L = 0
    for i in range(1, len(inds)):
        p1, p2 = pts[inds[i - 1]], pts[inds[i]]
        L += dist(p1, p2)
    return L + dist(p0, pts[inds[0]])

class Field:
    def __init__(self,width, height,number_points):
        self.width = width
        self.height = height
        self.number_points = number_points
        self.points = []
        self.difficultAreas = []

    def add_point(self, px, py):
        self.points.append((px, py))

    def add_points(self, *lst: tuple[int, int]):
        for p in lst:
            self.add_point(*p)

    def init_random_points(self, n, width, height):
        self.points = [[randint(0, width), randint(0, height)] for i in range(n)]

    def draw(self, screen):
        for p in self.points:
            pg.draw.circle(screen, (0,0,0), (p[0], p[1]), 1)
        for a in self.difficultAreas:
            a.draw(screen)


class DifficultArea:
    def __init__(self,x, y, a=1.0, b=1.0, alpha=0.0):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.alpha = alpha
    def draw(self, screen):
        it=self
        pg.draw.ellipse(screen, (0, 0, 0), (it.x-it.a/2, it.y-it.b/2, it.x+it.a/2, it.y+it.b/2),1)
    def hasPoint(self, x, y):
        return np.square((x-self.x)/self.a)+np.square((y-self.y)/self.b) < 1
    def hasPoint2(self, x, y):
        pass

class Individ:
    def __init__(self, route):
        self.fitness = 0
        self.route = route
        self.bestResult = 0
        self.meanResult = 0
        self.bestEpochResult = 0
    def calcFitness(self, p0, pts):
        self.fitness = getRouteLen(p0, pts, self.route)

class TSPCO:
    def __init__(self, numIndivids, numPoints):
        self.numIndivids = numIndivids
        self.population = []

    def caclFitnessPopulation(self, p0, pts):
        sumFitnessPopulation = 0
        bestResult = 0 # Лучшее значение особи с минимальным расстоянием
        for individ in self.population:
            individ.calcFitness(p0, pts)
            f = individ.fitness
            sumFitnessPopulation += f
            if f < bestResult:
                bestResult = f

def exchage(individ, i1, i2):
    # i1, i2 - числа от 1 до n, где n - число городов
    individ.route[i1-1], individ.route[i2-1] = individ.route[i2-1], individ.route[i1-1]


def inverse(individ, i1, i2):
    # i1, i2 (i1 < i2) - числа от 1 до n, где n - число городов
    if i2 < i1:
        i1, i2 = i2, i1

    di = (i2 - i1 + 1) // 2
    for i in range(di):
        individ.route[i1 + i - 1], individ.route[i2 - i - 1] = individ.route[i2 - i - 1], individ.route[i1 + i - 1]

def mutation(invivid, func, i1, i2):
    func(invivid, i1, i2)

def crossing(invdivid1, individ2, func, lst):
    pass

def test_exchange(i1, i2):
    class Ex:
        def __init__(self):
            self.route = [1,2,3,4,5]

    examp = Ex()

    mutation(examp, exchage, i1, i2)
    print(f"[1,2,3,4,5] with {i1} and {i2} exchage to {examp.route}")

def test_inverse(i1, i2):
    class Ex:
        def __init__(self):
            self.route = [1,2,3,4,5]

    examp = Ex()

    mutation(examp, inverse, i1, i2)
    print(f"[1,2,3,4,5] with {i1} and {i2} inverse to {examp.route}")

def tests_func(n, func_test):
    for i1 in range(1, n+1):
        for i2 in range(1, n+1):
            func_test(i1, i2)

if __name__ == '__main__':
    tests_func(5, test_exchange)
    tests_func(5, test_inverse)
