from random import randint, random, shuffle, choice, sample
from drawing import *
from matplotlib import pyplot as plt
from copy import deepcopy

def dist(p1, p2):
    """
    Функция вычисления расстояния между двумя точками
    :param p1: первая точка
    :param p2: вторая точка
    :return: расстояние. Тип np.float64
    """
    return np.linalg.norm(np.subtract(p1, p2))

def getRouteLen(p0, pts, route):
    """
    Функция для вычисления длины маршрута от исходного положения робота
    :param p0: положение робота
    :param pts: список координат точек
    :param route: список номеров последовательности посещения точек маршрута
    :return: расстояние пути маршрута в виде вещественного числа
    """
    L = 0
    for i in range(1, len(route)):
        p1, p2 = pts[route[i - 1]], pts[route[i]]
        L += dist(p1, p2)
    return L + dist(p0, pts[route[0]])

def getLineTime(p1, p2, listAreas):
    """
    Функция вычисления времени маршрута между двумя точками
    :param p1: первая точка [метры, метры]
    :param p2: вторая точка [метры, метры]
    :param listAreas: список областей с затруднённой проходимостью
    :return: время в секундах
    """
    func = findPointsIntersectionEllipseWithSegment
    intersects = []
    for e in listAreas:
        pIter = func(p1, p2, e.a, e.b, e.getPos(), e.alpha)
        intersects += pIter
    intersects.sort(key=lambda p: dist(p1, p))
    line = [p1] + intersects + [p2]
    mid_points = [getMedian(line[i - 1], line[i]) for i in range(1, len(line))]
    t = 0
    for j in range(1, len(line)):
        kList = [e.koef for e in listAreas if e.hasPoint(mid_points[j - 1])] + [1.0]
        v = min(kList)
        t += dist(line[j - 1], line[j]) / v

    return t

def getTimeRoute(p0, pts, route, listAreas):
    """
    Функция вычисления времени маршрута робота
    :param p0: исходное положение робота
    :param pts: список координат точек
    :param route: порядок посещения точек в соответствии с pts
    :param listAreas: список областей с затруднённой проходимостью
    :return: Время маршрута в секундах
    """
    T = 0
    iter, mids = [], []
    for i in range(1, len(route)):
        p1, p2 = pts[route[i - 1]], pts[route[i]]
        T += getLineTime(p1, p2, listAreas)

    return T + getLineTime(p0, pts[route[0]], listAreas)

def getLineTime_test(p1, p2, listAreas):
    """
    Функция тестирования вычисления времени маршрута между двумя точками
    :param p1: первая точка [метры, метры]
    :param p2: вторая точка [метры, метры]
    :param listAreas: список областей с затруднённой проходимостью
    :return: время в секундах, список точек пересечений областей,
    центры отрезков, образующихся точками пересечений и исходными точками маршрута
    """
    func = findPointsIntersectionEllipseWithSegment
    intersects = []
    for e in listAreas:
        pIter = func(p1, p2, e.a, e.b, e.getPos(), e.alpha)
        intersects += pIter
    intersects.sort(key=lambda p: dist(p1, p))
    line = [p1] + intersects + [p2]
    mid_points = [getMedian(line[i - 1], line[i]) for i in range(1, len(line))]
    t = 0
    for j in range(1, len(line)):
        kList = [e.koef for e in listAreas if e.hasPoint(mid_points[j - 1])] + [1.0]
        v = min(kList)
        t += dist(line[j - 1], line[j]) / v

    return t, intersects, mid_points

def getTimeRoute_test(p0, pts, route, listAreas):
    """
    Функция тестирования вычисления времени маршрута робота
    :param p0: исходное положение робота
    :param pts: список координат точек
    :param route: порядок посещения точек в соответствии с pts
    :param listAreas: список областей с затруднённой проходимостью
    :return: Время маршрута в секундах, список точек пересечений областей,
    центры отрезков, образующихся точками пересечений и исходными точками маршрута
    """
    T = 0
    iter, mids = [], []
    for i in range(1, len(route)):
        p1, p2 = pts[route[i - 1]], pts[route[i]]
        t, iter1, mids1 = getLineTime_test(p1, p2, listAreas)
        T += t
        iter += iter1
        mids += mids1
        # T += getLineTime(p1, p2, listAreas)

    t, iter1, mids1 = getLineTime_test(p0, pts[route[0]], listAreas)
    return T + t, iter + iter1, mids + mids1

def getDistMatrix(pts):
    """
    Функция получения матрицы расстояний между точками
    :param pts: список координат точек
    :return: двухмерная квадратная матрица в виде вложенных списков
    """
    numPoints = len(pts)
    return [[dist(pts[i], pts[j]) if i != j else 0.0 for i in range(numPoints)] for j in range(numPoints)]

def getTimeMatrix(pts, listAreas):
    """
    Функция получения матрицы времени маршрута между парами точек
    :param pts: список координат точек
    :param listAreas: список областей с затруднённой проходимостью
    :return: двухмерная квадратная матрица времени в виде вложенных списков
    """
    numPoints = len(pts)
    return [[getLineTime(pts[i], pts[j], listAreas) if i != j else 0.0 for i in range(numPoints)] for j in range(numPoints)]

def getRouteLen2(p0Dists, route, distMatrix):
    """
    Функция для вычисления длины маршрута на основе матрицы расстояний между точками маршрута
    :param p0Dists: список расстояний от положения робота до точек маршрута
    :param route: список последовательности посещения точек маршрута
    :param distMatrix: двухмерная квадратная матрица расстояний
    :return: расстояние пути маршрута от исходного положения робота
    """
    L = 0
    for i in range(1, len(route)):
       L += distMatrix[route[i-1]][route[i]]

    return L + p0Dists[route[0]]

def getTimeRoute2(p0Times, route, timeMatrix):
    """
    Функция для вычисления длины маршрута на основе матрицы времени маршрутов между двумя точками
    :param p0Times: список времени от положения робота до точек маршрута робота
    :param route: список последовательности посещения точек маршрута
    :param timeMatrix: двухмерная квадратная матрица времени
    :return: время пути маршрута от исходного положения робота
    """
    return getRouteLen2(p0Times, route, timeMatrix)

def getDistsFromPoint2Points(p0, pts):
    """
    Функция получения списка расстояний
    :param p0: исходная точка
    :param pts: список точек, p0 != p, где p in pts
    :return: Список расстояний
    """
    return [dist(p0, p) for p in pts]

def getTimesFromPoint2Points(p0, pts, listAreas):
    """
    Функция получения списка времени
    :param p0: исходная точка
    :param pts: список точек
    :param listAreas: список областей с затруднённой проходимостью
    :return: список времён
    """
    return [getLineTime(p0, p, listAreas) for p in pts]

class Field:
    """
    """
    def __init__(self,width, height, x=0, y=0):
        """

        :param width: Ширина среды моделирования
        :param height: высота среды моделирования
        :param x: горизонтальная координата левого верхнего угла среды моделирования
        :param y: вертикальная координата левого верхнего угла среды моделирования
        """
        self.width = width
        self.height = height
        self.points = [] #Список точек
        self.robot = None
        self.x = x
        self.y = y
        self.difficultAreas = [] # Список областей с затруднённой прохожимостью

    def initPoints(self, numPoints):
        """
        Функция инициализации точек среды
        :param numPoints: число точек
        """
        self.points = [[randint(self.x, self.x+self.width), randint(self.y, self.y + self.height)] for i in range(numPoints)]

    def initRobot(self, x0=None, y0=None):
        """
        Функция инициализации робота и его положения
        :param x0: горизонтальная координата робота относительно окна
        :param y0: вертикальная координата робота относительно окна

        Если x0=None,y0=None, то положение робота инициализируется случайно
        """
        if x0 is None or y0 is None:
            self.robot = Robot(randint(self.x, self.x +self.width), randint(self.y, self.y +self.height))
        else:
            self.robot = Robot(x0, y0)

    def initDifficultAreas(self, numAreas):
        """
        Инициализация областей с затруднёнными проходимотсями
        :param numAreas: число областей
        """
        rnd = randint
        xmin, xmax = self.x, self.x + self.width
        ymin, ymax = self.y, self.y + self.height
        amin, amax = 60, 300
        alpha_min, alpha_max = -180, 180
        self.difficultAreas = [DifficultArea(rnd(xmin, xmax), rnd(ymin, ymax),rnd(amin,amax), rnd(amin,amax),rnd(alpha_min, alpha_max)) for i in range(numAreas)]

    def draw(self, screen):
        """
        Рисование среды моделирования и её элементов
        :param screen: окно отрисовки
        :return:
        """
        for i, p in enumerate(self.points, 0):
            pg.draw.circle(screen, (0,0,0), (p[0], p[1]), 1)
            drawText(screen, f"{i + 1}", p[0], p[1])
        for a in self.difficultAreas:
            a.draw(screen)
        if self.robot is not None:
            self.robot.draw(screen)


class DifficultArea:
    def __init__(self,x, y, a=1.0, b=1.0, alpha=0.0):
        """
        Инициализация области с затруднённой проходимостью
        :param x: горизонтальная координата центра области
        :param y: горизонтальная координата центра области
        :param a: значениt главной полуоси области
        :param b: значение побочной полуоси области
        :param alpha: угол поворота области в градусах
        """
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.alpha = alpha
        self.koef = randint(20,80)/100 #Коэффициент проходимости
        self.color = (randint(0, 255), randint(0,255), randint(0,255))

    def getPos(self):
        """
        :return: положение центра области
        """
        return [self.x, self.y]

    def draw(self, screen):
        """
        Функция отрисовки области, области представляет собой эллипс
        :param screen: окно отрисовки
        """
        drawEllipse(screen, self.a, self.b, [self.x, self.y], self.alpha, self.color)
        pg.draw.circle(screen,self.color,[self.x,self.y],2)
        drawText(screen, f"k={self.koef:.2f}", self.x, self.y)

    def hasPoint(self, p):
        """
        Функция принадлежности точки к области
        :param p: координаты точки
        :return: True, если принадлежит, False, если не принадлежит
        """
        return inEllipse(p, self.a, self.b, [self.x, self.y], self.alpha)

class Robot:
    def __init__(self, x, y):
        """
        Инициализация робота
        :param x: горизонтальная координата робота
        :param y: вертикальная координата робота
        """
        self.x=x
        self.y=y
    def getPos(self):
        """

        :return: Положени робота
        """
        return [self.x, self.y]

    def draw(self, screen):
        """
        Функция отрисовки робота
        :param screen: окно отрисовки
        :return:
        """
        pg.draw.circle(screen, (0,255,255),self.getPos(),3, 2)
        drawText(screen, f"R", self.x, self.y)

class Individ:
    def __init__(self, route):
        """
        :param route: список порядка посещения пунктов на карте (пункты записаны в виде неотрицательных целых чисел)
        """
        self.fitness = -100500
        self.route = route
    def calcFitness(self, p0, pts):
        """
        Вычисление фитнес-функции индивида
        :param p0: положение робота
        :param pts: список положений точек посещения
        """
        self.fitness = -getRouteLen(p0, pts, self.route)

    def calcFitnessMatrix(self, p0Dists, distMatrix):
        """
        Вычисление фитнем-функции на основе списка и матрицы расстояний
        :param p0Dists: скписок расстнояний от p0 до остальных точек из distMatrix
        :param distMatrix: двухмерная квадратная матрица расстояний
        """
        self.fitness = -getRouteLen2(p0Dists, self.route, distMatrix)

    def __len__(self):
        return len(self.route)

    def __str__(self):
        return f"Individ(Route = {self.route}; Fitness = {self.fitness:.3f})"

class TSPCEA:
    """
    Класс для осуществления симуляциии работы эволюционного алгоритма оптимизациии задачи коммивояжера
    """
    POPULATION_SIZE = 200 # Размер популяции
    P_CROSSOVER = 0.9 # Вероятность скрещивания особей
    P_MUTATION = 0.1 # Веротяность мутации особи
    MAX_GENERATION = 500  # Максимальное число поколений
    def __init__(self, numIndivids, numPoints, numEliteIndivids=0):
        """

        :param numIndivids: число решений (особей)
        :param numPoints: число точек (пунктов назначения)
        :param numEliteIndivids: Число элитарных особей
        """
        self.numIndivids = numIndivids
        self.numPoints = numPoints
        self.population = []  # Список особей

        self.numGeneration = 0  # Номер эпохи
        self.maxFitness = -100500  # Максимальная пригодность популяции
        self.minFitness = 100500  # Минимальная пригодность популяции
        self.avgFitness = 0  # Средняя пригодность популяции
        self.maxFitnessList = []  # Список максимальных пригодностей популяции в поколение
        self.minFitnessList = []  # Список минимальных пригодностей популяции в поколение
        self.avgFitnessList = []  # Список средних пригодностей популяции в поколение
        self.numEliteIndivids = numEliteIndivids
        self.listEliteIndivids = [Individ(list(range(numPoints)))]*numEliteIndivids

    def caclFitnessPopulation(self, p0, pts):
        """
        Функция вычисления пригодности популяции

        :param p0: положение робота
        :param pts: положения точек в порядке их создания
        """
        sumFitnessPopulation = 0
        maxFitnessIndivid = -100500 # Лучшее значение особи с минимальным расстоянием
        minFitnessIndivid = 100500
        for individ in self.population:
            individ.calcFitness(p0, pts)
            fitness = individ.fitness
            if fitness > maxFitnessIndivid: maxFitnessIndivid = fitness
            if fitness < minFitnessIndivid: minFitnessIndivid = fitness
            sumFitnessPopulation += fitness

        # if minFitnessIndivid > self.minFitness: self.minFitness = minFitnessIndivid
        # if maxFitnessIndivid > self.maxFitness: self.maxFitness = maxFitnessIndivid
        self.minFitness = minFitnessIndivid
        self.maxFitness = maxFitnessIndivid
        self.avgFitness = sumFitnessPopulation/self.numIndivids

    def caclFitnessPopulation2(self, p0Dist, distMatrix):
        """
        Функция вычисления пригодности популяции при известной матрице расстояний

        К фитнес функции индивида добавляется
        :param p0: положение робота
        :param p1: положения точек в порядке их создания
        :param distMatrix: матрица расстояний
        """
        sumFitnessPopulation = 0
        maxFitnessIndivid = -100500 # Лучшее значение особи с минимальным расстоянием
        minFitnessIndivid = 100500
        for individ in self.population:
            # Вычисления фитнес функции особи, которая равна отрицательному расстоянию
            individ.calcFitnessMatrix(p0Dist,distMatrix)  # Вычисляет фитнес функцию индивида, которая равна отрицательному расттоянию
            fitness = individ.fitness
            if fitness > maxFitnessIndivid: maxFitnessIndivid = fitness
            if fitness < minFitnessIndivid: minFitnessIndivid = fitness
            sumFitnessPopulation += fitness

        self.minFitness = minFitnessIndivid
        self.maxFitness = maxFitnessIndivid
        self.avgFitness = sumFitnessPopulation / self.numIndivids

        """#Приведение фитнес-функций особей к положительным значениям
        print(maxFitnessIndivid, minFitnessIndivid, sumFitnessPopulation/self.numIndivids)
        for ind in self.population:
            ind.fitness = -(maxFitnessIndivid + minFitnessIndivid)

        print(self.population[-1].fitness)

        # if minFitnessIndivid > self.minFitness: self.minFitness = minFitnessIndivid
        # if maxFitnessIndivid > self.maxFitness: self.maxFitness = maxFitnessIndivid
        self.minFitness = -maxFitnessIndivid
        self.maxFitness = -minFitnessIndivid
        self.avgFitness = -sumFitnessPopulation/self.numIndivids - maxFitnessIndivid - minFitnessIndivid 
        #взято с "-", т. к. фитнес имеет "-"
        # В результате преобразования фитнес-функции особей имеют положительные значения"""

    def transformFitnessIndivids(self, maxFit, minFit):
        """
        Преобразование фитнес-функции особей на основе максимальной и минимальной приспособленности поколения
        В результате преобразования должны получится положительные фитнес-функции

        :param maxFit: максимальная присапособленность поколения
        :param minFit: минимальная приспособленность поколения
        """
        for ind in self.population:
            ind.fitness += -maxFit - minFit
        self.minFitness += -maxFit - minFit
        self.maxFitness += -maxFit - minFit
        self.avgFitness += -maxFit - minFit

    def chooseEliteIndivids(self, reverse=False):
        """
        Функция отбора элитных решений
        по умолчанию выбираются наиболее приспособленные особи
        :param reverse: Если True, то выбираются наихудшие особи, если False, то наилучшие
        :return:
        """
        genenrationEliteIndivids =  sorted(self.population, key=lambda indvd: -indvd.fitness, reverse=reverse)[:self.numEliteIndivids]
        self.listEliteIndivids = deepcopy(genenrationEliteIndivids)
        """k1, k2 = 0, 0
        newListEliteIndivids = []
        while len(newListEliteIndivids) < self.numEliteIndivids:
            if genenrationEliteIndivids[k1].fitness > self.listEliteIndivids[k2].fitness:
                newListEliteIndivids.append(deepcopy(genenrationEliteIndivids[k1]))
                k1 += 1
            else:
                newListEliteIndivids.append(deepcopy(genenrationEliteIndivids[k2]))
                k2 += 1

        self.listEliteIndivids = newListEliteIndivids"""
        #print(*[ind for ind in self.listEliteIndivids])

    def deleteDuplicates(self):
        """
        Функция удаления дубликатов
        """
        self.population.sort(key=lambda ind: ind.fitness)

        for i in range(1, self.numIndivids):
            ind1, ind2 = self.population[i-1], self.population[i]
            if ind1.route == ind2.route:
                mutationInverse(ind1)

        shuffle(self.population)

    def init(self):
        """
        Функция инициализации популяции
        Метод инициализации - случайный
        """
        self.population = randomInit(self.numIndivids, self.numPoints)

    def init2(self, numNearestNeighbour, distMatrix):
        """
        Функция инициализации популяции: метод случайный и на основе ближайшего соседа
        :param numNearestNeighbour: число особей, которые необходимо проинициализировать на основе ближайшего соседа
        :param distMatrix: матрица расстояний между пунктами
        """
        minNumInd = min(numNearestNeighbour, self.numIndivids)
        self.population = randomInit(self.numIndivids - minNumInd, self.numPoints)
        self.population += closestNeighbourInit(minNumInd, self.numPoints,distMatrix)

    def selection(self, numSelect, tourSize):
        """
        Функция отбора решений популяции.
        Метод отбора - турнир

        :param numSelect: Число отбираемых особей
        :param tourSize: Размер турнира
        :return: список отобранных особей
        """

        selected = tournament(self.population, numSelect, tourSize)
        #print(f"Gen: {self.numGeneration}, Len(selected) = {len(selected)}")
        return deepcopy(selected)

    def crossover(self, parents):
        """
        Функция скрещивания популяции
        Метод: упорядоченное скрещивание

        :param parents: Список особей для скрещивания
        :return: список новых решений
        """
        children = []
        numChild = self.numIndivids - self.numEliteIndivids
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            if len(children) >= self.numIndivids - self.numEliteIndivids: break
            if random() < TSPCEA.P_CROSSOVER:
                child1, child2 = crossoverOrdered(deepcopy(parent1), deepcopy(parent2))
                children.append(child1)
                children.append(child2)
            else:
                children.append(deepcopy(parent1))
                children.append(deepcopy(parent2))

        children = children[:numChild]

        return children

    def mutation(self, children):
        """
        Функция мутациии популяции
        Метод: Инверсия
        :param children: Список новых решений
        """
        for mutant in children:
            if random() < TSPCEA.P_MUTATION:
                mutationInverse(mutant)
                # mutationShuffleIndexes(mutant, 1.0/self.numPoints)

    def epoch(self, p0, pts):
        """
        Функция симуляции алгоритма эволюционной оптимизации

        :param p0: положение робота
        :param pts: положение точек
        """
        #print(f"Gen: {self.numGeneration}, before fitness")
        self.caclFitnessPopulation(p0, pts)
        #print(f"Gen: {self.numGeneration}, after fitness")
        self.trace()
        #print(f"Gen: {self.numGeneration}, Len(population) = {len(self.population)}")
        self.chooseEliteIndivids()

        parents = self.selection(self.numIndivids - self.numEliteIndivids, 2)
        #print(f"Gen: {self.numGeneration}, parents = ", *[p for p in parents])
        children = self.crossover(parents)
        #print(f"Gen: {self.numGeneration}, children = ",*[c for c in children])
        self.mutation(children)

        #print(f"Gen: {self.numGeneration}, Len(children) = {len(children)}")
        self.population = children + self.listEliteIndivids
        #self.caclFitnessPopulation(p0, pts)
        #print(f"Gen: {self.numGeneration}, Len(population) = {len(self.population)}")

    def epoch2(self, p0Dist, distMatrix):
        """
        Функция симуляции алгоритма эволюционной оптимизации на основе матрицы и списка расстояний/времен
        :param p0Dist: Список расстояний/времен от положения робота
        :param distMatrix: матрица расстояний/времен
        """
        #print(f"Gen: {self.numGeneration}, before fitness")
        self.caclFitnessPopulation2(p0Dist, distMatrix)
        #self.transformFitnessIndivids(self.maxFitness, self.minFitness)
        #print(f"Gen: {self.numGeneration}, after fitness")
        self.trace()
        #print(f"Gen: {self.numGeneration}, Len(population) = {len(self.population)}")
        self.chooseEliteIndivids()

        parents = self.selection(self.numIndivids - self.numEliteIndivids, 2)
        #print(f"Gen: {self.numGeneration}, parents = ", *[p for p in parents])
        children = self.crossover(parents)
        #print(f"Gen: {self.numGeneration}, children = ",*[c for c in children])
        self.mutation(children)

        #print(f"Gen: {self.numGeneration}, Len(children) = {len(children)}")
        self.population = children + self.listEliteIndivids
        #self.deleteDuplicates()
        #print(f"Gen: {self.numGeneration}, Len(population) = {len(self.population)}")


    def trace(self):
        """
        Функция записи характеристик популяции
        """
        self.maxFitnessList.append(self.maxFitness)
        self.minFitnessList.append(self.minFitness)
        self.avgFitnessList.append(self.avgFitness)
        self.numGeneration += 1

    def getGraphics(self,filename):
        """
        Функция получения графиков характеристик популяции
        """
        epochs = list(range(0, self.numGeneration))
        plt.plot(epochs, self.maxFitnessList, label="Макс")
        plt.plot(epochs, self.minFitnessList, label="Мин")
        plt.plot(epochs, self.avgFitnessList, label="Среднее")
        plt.xlabel("Поколение")
        plt.ylabel("Фитнес-функция")
        plt.title("График макс,мин и средней фитнес-функции поколений")
        plt.legend()
        plt.savefig(filename)
        plt.show()

def randomInit(numIndivids, sizeIndivid):
    """
    Фунция случайно инициализации особей
    :param numIndivids: число особей
    :param sizeIndivid: размер особи
    :return: список особей
    """
    population = []
    for i in range(numIndivids):
        inds = list(range(0, sizeIndivid))
        shuffle(inds)
        population.append(Individ(inds))

    return population

def closestNeighbourInit(numIndivids, sizeIndivid, distMatrix, init=-1):
    """
    Функция инициализации особей на основе ближайшего соседа
    :param numIndivids: число особей
    :param sizeIndivid: размер особей
    :param distMatrix: матрица расстояний между пунтками
    :param init: параметр для проверки и исследования работы алгоритма,
    если init < 0, то первый пункт выбирается случайно,
    иначе устанавливается равным init
    :return: список особей
    """
    population = []
    for i in range(numIndivids):
        if init < 0:
            next = randint(0, sizeIndivid - 1)
        else:
            next = init % sizeIndivid
        route = [next]
        while len(route) < sizeIndivid:
            dst, next = min([[d, j] for j, d in enumerate(distMatrix[next]) if j not in route], key=lambda ind: ind[0])
            route.append(next)

        population.append(Individ(route))

    return population

def closest2NeighbourInit(numIndivids, sizeIndivid, distMatrix, init=-1):
    """
    Функция инициализации особей на основе ближайших двух соседей
    :param numIndivids: число особей
    :param sizeIndivid: размер особей
    :param distMatrix: матрица расстояний между пунтками
    :param init: параметр для проверки и исследования работы алгоритма,
    если init < 0, то первый пункт выбирается случайно,
    иначе устанавливается равным init
    :return: список особей
    """
    population = []
    for i in range(numIndivids):
        if init < 0:
            next = randint(0, sizeIndivid - 1)
        else:
            next = init % sizeIndivid
        route = [next]
        while len(route) < sizeIndivid:
            neighbours = {j: d for j, d in enumerate(distMatrix[next]) if j not in route}
            #print(len(neighbours), *neighbours, neighbours.values())
            if len(neighbours) > 1:
                for neighbour in neighbours.keys():
                    neighbours[neighbour] += min([d for k, d in enumerate(distMatrix[neighbour]) if k not in route and k != neighbour])

            #print(len(neighbours), *neighbours, neighbours.values())

            next = min(neighbours, key=lambda k: neighbours[k])
            #print(next, neighbours[next])
            route.append(next)

        population.append(Individ(route))

    return population

"""def closest2NeighbourInit_beta(numIndivids, sizeIndivid, distMatrix, init=-1):
    
    Функция инициализации особей на основе ближайших двух соседей
    :param numIndivids: число особей
    :param sizeIndivid: размер особей
    :param distMatrix: матрица расстояний между пунтками
    :param init: параметр для проверки и исследования работы алгоритма,
    если init < 0, то первый пункт выбирается случайно,
    иначе устанавливается равным init
    :return: список особей
    
    population = []
    dist2Matrix = [[[distMatrix[i][j] + distMatrix[j][k] if i != j and j != k and i != k else 0.0 
                     for k in range(sizeIndivid)] 
                    for j in range(sizeIndivid)] 
                   for i in range(sizeIndivid)]
    for i in range(numIndivids):
        if init < 0:
            next = randint(0, sizeIndivid - 1)
        else:
            next = init % sizeIndivid
        route = [next]
        while len(route) < sizeIndivid:
            mat = dist2Matrix[next]
            dst, next = min([[d, j] for j, d in enumerate() if j not in route], key=lambda ind: ind[0])
            route.append(next)

        population.append(Individ(route))

    return population"""

def randomSelect(listIndivids: list[Individ], numSelectIndivids: int):
    """
    Функция случайного отбора особей
    :param listIndivids: список особей
    :param numSelectIndivids: число отбираемых особей
    :return: список отобранных особей
    """
    return [choice(listIndivids) for i in range(numSelectIndivids)]

def tournament(listIndivids: list[Individ], numSelectIndivids: int, sizeTournament: int):
    """
    Функция отбора на основе турнира
    :param listIndivids: список особей
    :param numSelectIndivids: число отбираемых особей
    :param sizeTournament: размер турнира
    :return: список отобранных особей
    """
    chosen = []
    for i in range(numSelectIndivids):
        participants = randomSelect(listIndivids, sizeTournament)
        winnerTournament = max(participants, key=lambda individ: individ.fitness)
        chosen.append(winnerTournament)

    return chosen

def rouletteSelect(listIndivids: list[Individ], numSelectIndivids: int):
    """
    Функция отбора на основе метода рулетки
    :param listIndivids: список особей
    :param numSelectIndivids: число отбираемых особей
    :return: список особей
    """
    sortedListIndivids = sorted(listIndivids, key=lambda ind: ind.fitness, reverse=True)
    sum_fits = sum(ind.fitness for ind in sortedListIndivids)
    #max_fit = -max(sortedListIndivids, key = lambda ind: ind.fitness)
    #min_fit = -min(sortedListIndivids, key = lambda ind: ind.fitness)
    chosen = []
    for i in range(numSelectIndivids):
        u = random() * sum_fits
        summa = 0
        for ind in sortedListIndivids:
            summa += ind.fitness
            if summa > u:
                chosen.append(ind)
                break

    return chosen

def mutationShuffleIndexes(individ: Individ, individPb: float):
    """
    Функция мутации особи на основе обмена значений
    :param individ: особь
    :param individPb: вероятность изменения координаты решения особи
    :return: Видоизмененнное решение
    """
    size = len(individ)
    individ.fitness = -100500
    for i in range(size):
        if random() < individPb:
            swap_index = randint(0, size - 2)
            if swap_index >= i:
                swap_index += 1
            individ.route[i], individ.route[swap_index] = \
            individ.route[swap_index], individ.route[i]

def mutationInverse(individ: Individ):
    """
    Функция мутации особи методом инверсии
    :param individ: особь
    """
    size = len(individ)
    if size == 0:
        return

    i1, i2 = randint(0,size-1), randint(0,size)
    left, right = min(i1, i2), max(i1, i2)

    individ.route[left:right] = individ.route[left:right][::-1]

def crossoverOrdered(individ1: Individ, individ2: Individ):
    """
    Функция срещивания особей на основе метода упорядоченного скрещивания
    :param individ1: первая особь
    :param individ2: вторая особь
    :return: список из двух новых особей
    """
    ind1, ind2 = deepcopy(individ1), deepcopy(individ2)

    size = min(len(ind1), len(ind2))
    left, right = sample(range(size), 2)
    if left > right:
        left,right = right,left
    hole1, hole2 = [True]*size, [True]*size
    for i in range(size):
        if i < left or i > right:
            hole1[ind2.route[i]] = False
            hole2[ind1.route[i]] = False

    # print (hole1, ind1.route, hole2, ind2.route)

    temp1, temp2 = ind1.route, ind2.route
    k1, k2 = right + 1, right + 1
    for i in range(size):
        if not hole1[temp1[(i + right + 1) % size]]:
            ind1.route[k1 % size] = temp1[(i + right + 1) % size]
            k1 += 1

        if not hole2[temp2[(i + right + 1) % size]]:
            ind2.route[k2 % size] = temp2[(i + right + 1) % size]
            k2 += 1

        # Swap the content between a and b (included)
    for i in range(left, right + 1):
        ind1.route[i], ind2.route[i] = ind2.route[i], ind1.route[i]

    ind1.fitness = -100500
    ind2.fitness = -100500
    return ind1, ind2


"""def exchage(individ, sizeIndivid):
    i1 = randint(0, sizeIndivid - 2)
    i2 = randint(1, sizeIndivid - 1)
    if i1 == i2: i1 -= 1
    # i1, i2 - числа от 0 до n-1, где n - число городов
    individ.route[i1], individ.route[i2] = individ.route[i2], individ.route[i1]

def mutation(invivid, func, i1, i2):
    func(invivid, i1, i2)

def crossover(individ1, individ2, func, *params):
    return func(individ1, individ2)"""

"""
def crossoverPartiallyMatched(individ1, individ2, ):
    i1 = np.random.randint(0, len(self.population) - 1)
    i2 = np.random.randint(i1 + 1, len(self.population))
    i = np.random.randint(0, self.numPoints)
    l1, r1 = individ1.route[:i], individ1.route[i:]
    l2, r2 = individ2.route[:i], individ2.route[i:]
    new_route1, new_route2 = l1 + r2, l2 + r1

    def correct_route(route):
        n = len(route)
        points = list(range(1, n+1))
        for i in range(n):
            point = route[i]
            if route.count(point) > 1:
                for an_point in points:  # подстановка для устранения коллизии
                    if an_point not in route:
                        route[i] = an_point
                        break

    correct_route(new_route1)
    correct_route(new_route2)

    return Individ(new_route1), Individ(new_route2)"""

"""def test_exchange(i1, i2):
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
"""
#if __name__ == '__main__':
    # tests_func(5, test_exchange)
    # tests_func(5, test_inverse)

"""def findBestRouteRandom(p0, pts):
    LBest=100500
    inds=None
    for i in range(1000):
        inds2=list(range(len(pts)))
        np.random.shuffle(inds2)
        L=getRouteLen(p0, pts, inds2)
        if L<LBest:
            LBest=L
            inds=inds2
    return inds, LBest

def findBestRouteFullSearch(p0, pts):
    routeBest=None
    LBest=100500
    import itertools
    z = itertools.permutations(list(range(len(pts))))
    i=0
    for ii in z:
        if i% 100000==0: print(i)
        L = getRouteLen(p0, pts, ii)
        if L<LBest:
            LBest=L
            routeBest=ii
        i+=1

    return routeBest, LBest"""

if __name__ == '__main__':
    pass