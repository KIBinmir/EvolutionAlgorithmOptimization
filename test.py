from drawing import *
from evolution_comminvoyadger import *

def test_intersection():
    """
    Тестирование функции нахождения точек пересечения отрезка и эллипса.
    Меняется положения отрезка и угол поворота эллипса
    Итог теста: завершён успешно
    """
    size = (800, 600)
    fps = 60
    screen = pg.display.set_mode(size)
    timer = pg.time.Clock()
    alpha = 0
    p0 = [400, 300]
    a = 200
    b = 100
    pts = [[200, 200], [500, 500]]
    p0 = [400, 400]
    ind = 1
    state1 = inEllipse(pts[0], a, b, p0, alpha)
    state2 = inEllipse(pts[1], a, b, p0, alpha)
    intersectPoints = findPointsIntersectionEllipseWithSegment(pts[0], pts[1], a, b, p0, alpha)

    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_1:
                    ind = 0
                if ev.key == pg.K_2:
                    ind = 1
                if ev.key == pg.K_w:
                    pts[ind][1] -= 10
                if ev.key == pg.K_s:
                    pts[ind][1] += 10
                if ev.key == pg.K_a:
                    pts[ind][0] -= 10
                if ev.key == pg.K_d:
                    pts[ind][0] += 10
                if ev.key == pg.K_e:
                    alpha += 5
                if ev.key == pg.K_q:
                    alpha -= 5
                state1 = inEllipse(pts[0], a, b, p0, alpha)
                state2 = inEllipse(pts[1], a, b, p0, alpha)
                intersectPoints = findPointsIntersectionEllipseWithSegment(pts[0], pts[1], a, b, p0, alpha)

        dt = 1 / fps




        screen.fill((255, 255, 255))
        drawEllipse(screen, a, b, p0, alpha, (0, 0, 40, 128), w=1, fill=True)

        drawText(screen, f"Points in ellipse = {[state1, state2]}", 5, 5)
        drawText(screen, f"Points = {pts}", 5, 25)
        drawText(screen, f"Angle = {alpha}", 5, 45)
        drawText(screen, f"Active point: {ind+1}", 5, 65)

        pg.draw.circle(screen, (255, 0, 0), pts[0], 3)
        pg.draw.circle(screen, (0, 255, 0), pts[1], 3)
        pg.draw.line(screen, (0, 0, 0), pts[0], pts[1], 1)
        if len(intersectPoints) > 0:
            drawText(screen, f"Intersection point 1: [{intersectPoints[0][0]:.0f},{intersectPoints[0][1]:.0f}]", 5, 85)
            if len(intersectPoints) > 1:
                drawText(screen, f"Intersection point 2: [{intersectPoints[1][0]:.0f},{intersectPoints[1][1]:.0f}]", 5,
                         105)
            for p in intersectPoints:
                pg.draw.circle(screen, (0, 255, 255), p, 3)


        pg.display.flip()
        timer.tick(fps)

def test_ellipse():
    """
    Тестирование отрисовки эллипса и принадлежности точки области эллипса.
    Меняется положение точки и угол поворота эллипса
    Итог теста: завершён успешно
    """
    size = (800, 600)
    fps = 60
    screen = pg.display.set_mode(size)
    timer = pg.time.Clock()
    alpha = 45
    a = 200
    b = 100
    p = [400, 400]
    p0 = [400, 400]
    state = inEllipse(p, a, b, p0, alpha)

    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_w:
                    p[1] -= 10
                if ev.key == pg.K_s:
                    p[1] += 10
                if ev.key == pg.K_a:
                    p[0] -= 10
                if ev.key == pg.K_d:
                    p[0] += 10
                if ev.key == pg.K_e:
                    alpha += 5
                if ev.key == pg.K_q:
                    alpha -= 5
                state = inEllipse(p, a, b, p0, alpha)

        dt = 1 / fps


        screen.fill((255, 255, 255))
        drawEllipse(screen, a, b, p0, alpha, (0,0,40, 128), w=1, fill=True)
        drawText(screen, f"Point in ellipse = {state}", 5, 5)
        drawText(screen, f"Point = {p}", 5, 25)
        drawText(screen, f"Angle = {alpha}", 5, 45)
        pg.draw.circle(screen, (255,0,0), p, 3)

        pg.display.flip()
        timer.tick(fps)

def test_drawRoute():
    """
    Тестирование отрисовки маршрута.
    Корректируются цвет линии, толщина линии, размеры точек маршрута

    Итог теста: завершён успешно
    """
    size = (800, 600)
    screen = pg.display.set_mode(size)
    timer = pg.time.Clock()
    fps = 20

    p0 = [100, 100]
    pts = [
        [200, 200],
        [200, 300],
        [300, 200],
        [400, 250],
        [350, 300],
        [400, 150],
        [200, 250],
        [500, 500],
        [100, 450]
    ]
    route = list(range(len(pts)))

    indsBest, LBest = None, None

    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_r:
                    shuffle(route)
                    # indsBest, LBest = findBestRouteRandom(p0, pts)

        dt = 1 / fps

        screen.fill((255, 255, 255))
        drawRoute(screen, p0, pts, route)

        pg.display.flip()
        timer.tick(fps)

def test_mutationInverse(n):
    """
    Тестирования операции мутации особи на ошибки

    :param n: ращмер особи

    Итог теста: завершён успешно
    """
    for i in range(n):
        ind = randomInit(1, 10)[0]
        print(ind)
        mutationInverse(ind)
        print(ind)

def test_closestNeghbourInit():
    """
    Тестирования методов инициализации на основе ближайших соседей и сравнение друг с другом.

    Результаты теста: завершён успешно
    Вывод из результатов: метод на основе иницализации ближайшего соседа чаще даёт наиболее короткие расстояния,
    чем метод инициализации на основе ближайших двух соседей
    """
    size = (1200, 900)
    screen = pg.display.set_mode(size)
    timer = pg.time.Clock()
    fps = 20

    """p0 = [100, 100]
    pts = [
        [200, 200],
        [200, 300],
        [300, 200],
        [400, 250],
        [350, 300],
        [400, 150],
        [200, 250],
        [500, 500],
        [100, 450]
    ]"""
    numPoints = 30
    field = Field(800, 600, 200, 150)
    field.initRobot()
    field.initPoints(numPoints)
    matrix = getDistMatrix(field.points)
    i=0
    ind = closestNeighbourInit(1, numPoints, matrix, init=i)[0]
    route = ind.route
    ind.calcFitness2(field.robot.getPos(), field.points[route[0]],matrix)

    ind2 = closest2NeighbourInit(1, numPoints, matrix, init=i)[0]
    route2 = ind2.route
    ind2.calcFitness2(field.robot.getPos(), field.points[route[0]], matrix)


    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_r:
                    i = (i + 1) % numPoints
                    ind = closestNeighbourInit(1, numPoints, matrix, init=i)[0]
                    route = ind.route
                    ind.calcFitness2(field.robot.getPos(), field.points[route[0]],matrix)

                    ind2 = closest2NeighbourInit(1, numPoints, matrix, init=i)[0]
                    route2 = ind2.route
                    ind2.calcFitness2(field.robot.getPos(), field.points[route[0]], matrix)
                if ev.key == pg.K_e:
                    i = (i - 1) % numPoints
                    ind = closestNeighbourInit(1, numPoints, matrix, init=i)[0]
                    route = ind.route
                    ind.calcFitness2(field.robot.getPos(), field.points[route[0]],matrix)

                    ind2 = closest2NeighbourInit(1, numPoints, matrix, init=i)[0]
                    route2 = ind2.route
                    ind2.calcFitness2(field.robot.getPos(), field.points[route[0]], matrix)

        dt = 1 / fps

        screen.fill((255, 255, 255))
        drawRoute(screen, field.robot.getPos(), field.points, route)
        drawRoute(screen, field.robot.getPos(), field.points, route2,color=(255,0,0))
        field.draw(screen)
        drawText(screen, f"route = {[i+1 for i in route]}", 5, 5)
        drawText(screen, f"DistClosest = {-ind.fitness}", 5, 25)
        drawText(screen, f"route2 = {[i + 1 for i in route2]}", 5, 45)
        drawText(screen, f"Dist2Closest = {-ind2.fitness}", 5, 65)

        pg.display.flip()
        timer.tick(fps)

def test_calcFitnessIndivid():
    """
    Тестирование методов рассчётов длин маршрутов
    без матрицы расстояний и с матрицей расстояний

    Результаты тестирования: завершён успешно
    Вывод из результатов: Расстояния вычисляется одинаково
    """
    sz = (1200, 900)
    screen = pg.display.set_mode(sz)
    timer = pg.time.Clock()
    fps = 20

    # p0=[100, 100]
    """pts=[
        [200,200],
        [200,300],
        [300,200],
        [400,250],
        [350,300],
        [400,150],
        [200,250],
        [500,500],
        [100,450]
         ]"""
    # pts = [[randint(0, sz[0]), randint(0, sz[1])] for i in range(30)]
    field = Field(800, 600, 200, 150)
    field.initRobot(200, 200)
    field.initPoints(29)
    matrix = getDistMatrix(field.points)
    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
    ind = randomInit(1, 29)[0]
    l1 = getRouteLen(field.robot.getPos(), field.points, ind.route)
    l2 = getRouteLen2(p0Dists, ind.route, matrix)

    while True:
        for ev in pg.event.get():
            if ev.type==pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_r:
                    ind = randomInit(1, 29)[0]
                    l1 = getRouteLen(field.robot.getPos(), field.points, ind.route)
                    l2 = getRouteLen2(p0Dists, ind.route, matrix)

        dt=1/fps

        screen.fill((255, 255, 255))
        field.draw(screen)

        drawText(screen, f"d1 = {l1}", 5, 5)
        drawText(screen, f"d2 = {l2}", 5, 25)
        drawRoute(screen, field.robot.getPos(), field.points,ind.route)

        pg.display.flip()
        timer.tick(fps)

def test_DifficultAreas():
    """
    Тестирования расчёта времени и расстояния на основе методов
    без использования матриц хранения расчётов между парами точек
    и с использованием матриц хранения расчётов между парами точек


    Результаты тестирования: завершено успешно
    Вывод из результатов: методы расчёта в рамках одной характеристики совпадают.
    При вычислении маршрута, при котором робот движется со скоростью v=1,
    на пути которого нет сред с затруднённой проходимостью,
    длина маршрута совпадает со временем маршрута

    Замечания: при расчёте времени маршрута была допущена ошибка,
    при котором появлялись точки пересечения с эллипсами вне отрезка между двумя точками
    Текущее состояние ошибки: устранена

    Для проверки используется функция, которая вычисляет время и одновременно
    возвращает все точки пересечения с эллипсами, а также точки между пересечениями
    """
    sz = (1200, 900)
    screen = pg.display.set_mode(sz)
    timer = pg.time.Clock()
    fps = 20

    # p0=[100, 100]
    """pts=[
        [200,200],
        [200,300],
        [300,200],
        [400,250],
        [350,300],
        [400,150],
        [200,250],
        [500,500],
        [100,450]
         ]"""
    # pts = [[randint(0, sz[0]), randint(0, sz[1])] for i in range(30)]
    numPoints =2
    field = Field(800, 600, 200, 150)
    field.initRobot(200, 200)
    field.initPoints(numPoints)
    field.initDifficultAreas(5)

    distMatrix = getDistMatrix(field.points)
    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
    ind = randomInit(1, numPoints)[0]
    dst = getRouteLen(field.robot.getPos(), field.points, ind.route)
    time, itersects, mids = getTimeRoute_test(field.robot.getPos(), field.points, ind.route, field.difficultAreas)
    dst2 = getRouteLen2(p0Dists, ind.route, distMatrix)
    time2 = getTimeRoute2(p0Times, ind.route, timeMatrix)
    i=0

    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_r:
                    ind = randomInit(1, numPoints)[0]
                    dst = getRouteLen(field.robot.getPos(), field.points, ind.route)
                    time = getTimeRoute(field.robot.getPos(), field.points, ind.route, field.difficultAreas)
                    dst2 = getRouteLen2(p0Dists, ind.route, distMatrix)
                    time2 = getTimeRoute2(p0Times, ind.route, timeMatrix)
                if ev.key == pg.K_d:
                    ind = closestNeighbourInit(1, numPoints, distMatrix, init=i)[0]
                    dst = getRouteLen(field.robot.getPos(), field.points, ind.route)
                    #time = getTimeRoute(field.robot.getPos(), field.points, ind.route, field.difficultAreas)
                    time, itersects, mids = getTimeRoute_test(field.robot.getPos(), field.points, ind.route,
                                                              field.difficultAreas)
                    dst2 = getRouteLen2(p0Dists, ind.route, distMatrix)
                    time2 = getTimeRoute2(p0Times, ind.route, timeMatrix)
                if ev.key == pg.K_t:
                    ind = closestNeighbourInit(1, numPoints, timeMatrix, init=i)[0]
                    dst = getRouteLen(field.robot.getPos(), field.points, ind.route)
                    #time = getTimeRoute(field.robot.getPos(), field.points, ind.route, field.difficultAreas)
                    time, itersects, mids = getTimeRoute_test(field.robot.getPos(), field.points, ind.route,
                                                              field.difficultAreas)
                    dst2 = getRouteLen2(p0Dists, ind.route, distMatrix)
                    time2 = getTimeRoute2(p0Times, ind.route, timeMatrix)
                if ev.key == pg.K_i:
                    i = (i + 1) % numPoints
                if ev.key == pg.K_s:
                    pg.image.save(screen, 'test_dif_areas.png')

        dt = 1 / fps

        screen.fill((255, 255, 255))
        field.draw(screen)

        #drawText(screen, f"dist = {dst:.0f} [m]", 5, 5)
        #drawText(screen, f"time = {time:.2f} [c]", 5, 25)
        #drawText(screen, f"time = {(time/60):.2f} [min]", 5, 45)
        drawText(screen, f"dist2 = {dst2:.0f} [m]", 5, 65)
        drawText(screen, f"time2 = {time2:.2f} [c]", 5, 85)
        drawText(screen, f"time2 = {(time2/60):.2f} [min]", 5, 105)
        drawRoute(screen, field.robot.getPos(), field.points, ind.route)
        for intr in itersects:
            pg.draw.circle(screen, (0,0,0), intr,3)
        for m in mids:
            pg.draw.circle(screen, (120,120,120), m,3)


        pg.display.flip()
        timer.tick(fps)

def test_TSPCEA():
    """
    Тестирования работы поиска оптимального маршрута с учётом сред с переменной проходимостью

    Результаты тестирования: завершено успешно
    Выводы из результатов:
    оптимальное число особей - 200,
    число элитарных особей 20,
    инициализация особей: случайная
    функция пригодности: -t, t - время маршрута
    метод скрещивания: упорядоченное скрещивание
    метод мутации: инверсия особи
    метод отбора: турнир из 2-х особей
    Число поколений, при котором достигается оптимальный результат:
    [Число точек, число поколений]
    [50, 1000]
    [75, 1500]
    [100, 2000]

    Замечания: При инициализации на основе ближайшего соседа происходит очень быстрая сходимость решения
    Удаление дупликатов особей не меняет время сходимости решения
    Мутация на основе перетасовки (взаимообмена) работает хуже метода инверсии
    """
    sz = (1200, 900)
    screen = pg.display.set_mode(sz)
    timer = pg.time.Clock()
    fps = 20

    # p0=[100, 100]
    """pts=[
        [200,200],
        [200,300],
        [300,200],
        [400,250],
        [350,300],
        [400,150],
        [200,250],
        [500,500],
        [100,450]
         ]"""
    # pts = [[randint(0, sz[0]), randint(0, sz[1])] for i in range(30)]
    numPoints = 50
    numAreas = 0
    field = Field(800, 600, 200, 150)
    field.initRobot()
    field.initPoints(numPoints)
    field.initDifficultAreas(numAreas)
    # route=list(range(0, len(pts)))

    distMatrix = getDistMatrix(field.points)
    p0Dists = getDistsFromPoint2Points(field.robot.getPos(),field.points)
    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
    p0Times = getTimesFromPoint2Points(field.robot.getPos(),field.points, field.difficultAreas)

    routeBest, timeBest=None, None
    dst = 0
    ftnss = 0

    ga = TSPCEA(200, numPoints, numEliteIndivids=10)
    ga.init()
    ga.caclFitnessPopulation2(p0Times, timeMatrix)

    while True:
        for ev in pg.event.get():
            if ev.type==pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_g:
                    for i in range(100):
                        #ga.epoch(field.robot.getPos(), field.points)
                        ga.epoch2(p0Times, timeMatrix)
                        routeBest=ga.listEliteIndivids[0].route
                        timeBest=getTimeRoute2(p0Times, routeBest, timeMatrix)
                        dst = getRouteLen2(p0Dists, routeBest, distMatrix)
                        ftnss = ga.listEliteIndivids[0].fitness

                        print(f"Generation: {ga.numGeneration}")
                        print(f"Best route: {routeBest}")
                        print(f"Time: {timeBest}")
                        print(f"Dist: {dst}")
                        print(f"Fitness: {ftnss}")

                    # print(*[ind.fitness for ind in ga.listEliteIndivids])
                if ev.key == pg.K_0:
                    numAreas = 0
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest=None
                if ev.key == pg.K_1:
                    numAreas=1
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_2:
                    numAreas=2
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_3:
                    numAreas=3
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_4:
                    numAreas=4
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_5:
                    numAreas=5
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None

                if ev.key == pg.K_s:
                    ga.getGraphics("test_graphic50.png")
                    pg.image.save(screen, 'test_field50.png')

                if ev.key == pg.K_c:
                    ga = TSPCEA(200, numPoints, numEliteIndivids=10)
                    ga.init()
                    ga.caclFitnessPopulation2(p0Times, timeMatrix)
                    routeBest = None

        dt=1/fps

        screen.fill((255, 255, 255))
        field.draw(screen)

        # L=getRouteLen(p0, pts, route)

        # drawRoute(screen, p0, pts, route)

        # drawText(screen, f"inds = {route}", 5, 5)
        # drawText(screen, f"L = {L:.2f}", 5, 25)
        drawText(screen, f"Generation = {ga.numGeneration}", 5, 5)
        if routeBest is not None:
            drawText(screen, f"BestRoute = {[i + 1 for i in routeBest]}", 5, 25)
            drawText(screen, f"TimeBest = {timeBest:.2f} [sec]", 5, 45)
            drawText(screen, f"dist = {dst:.0f} [m]", 5, 65)
            drawText(screen, f"fitness = {ftnss:.0f}", 5, 85)

            drawRoute(screen, field.robot.getPos(), field.points, routeBest, (255, 0, 0))


        pg.display.flip()
        timer.tick(fps)

if __name__ == '__main__':
    # test_intersection()
    # test_ellipse()
    # test_drawRoute()
    # test_mutationInverse(50)
    #test_calcFitnessIndivid()
    # test_closestNeghbourInit()
    # test_DifficultAreas()
    test_TSPCEA()