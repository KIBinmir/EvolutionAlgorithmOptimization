import pygame as pg
import numpy as np
import sys

pg.font.init()
font = pg.font.SysFont('Comic Sans MS', 20)
def drawText(screen, s, x, y):
    """
    Функция отрисовки текста на окне
    Для работы необходимо проинициализировать шрифт в pygame
    :param screen: окно отрисовки
    :param s: текст
    :param x: горизонтальная координата относительно верхнего левого угла, отсчёт идёт вправо
    :param y: вертикальная координата относительно верхнего левого угла, остчёт идёт вниз
    """
    surf=font.render(s, True, (0,0,0))
    screen.blit(surf, (x,y))

def limAng(ang):
    """
    Функция ограничения угла до диапазона [-pi, pi]
    :param ang: угол в радианах
    :return: угол в диапазоне [-pi, pi]
    """
    while ang > np.pi: ang -= 2 * np.pi
    while ang <= -np.pi: ang += 2 * np.pi
    return ang

def rot(v, ang):
    """
    функция поворота вектора
    :param v: вектор в декартновых координатах [x, y]
    :param ang: угол поворота относительно центра координат [0, 0]
    :return: вектор, повёрнутый на угол ang
    """
    s, c = np.sin(ang), np.cos(ang)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]

def addVec(v, v1):
    """
    Функция суммы векторов
    :param v: первый вектор [x, y]
    :param v1: второй вектор [x1, y1]
    :return: сумма векторов в формате [x, y]
    """
    return [v[0] + v1[0], v[1] + v1[1]]

def getMedian(p1, p2):
    """
    Функция вычсиления середины отрезка
    :param p1: первый конец отрезка [x1, y1]
    :param p2: второй конец отрезка [x2, y2]
    :return: середина отрезка [x, y]
    """
    return [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2]

def subVec(v, v1):
    """
    Функция разницы векторов
    :param v: первый вектор [x, y]
    :param v1: второй вектор [x1, y1]
    :return: разница векторов в формате [x, y]
    """
    return [v[0] - v1[0], v[1] - v1[1]]

def inEllipse(p, a, b, p0, alpha):
    """
    Функция принадлежности точки к эллипсу
    :param p: координаты точки типа [x, y]
    :param a: длина главной полуоси
    :param b: длина побочной полуоси
    :param p0: центр эллипса [x0,y0]
    :param alpha: угол поворота осей отностельно оси X
    :return: True - если принадлежит, False - если не принадлежит
    """
    alpha_rad = np.pi * alpha / 180
    x1, y1 = p[0] - p0[0], p[1] - p0[1]
    x2, y2 = rot([x1, y1], -alpha_rad)
    return x2*x2/a/a + y2*y2/b/b <= 1.0

def inSegmentLine(p, p1, p2):
    """
    Функция принадлежности точки отрезку

    Замечание: функция используется для точек, которые лежат на прямой,
    но могут не принадлежать отрезку между точками, образующими прямую
    :param p: точка, принадлежащая прямой
    :param p1: первая точка, образующая отрезок
    :param p2: вторая точка, образующая отрезок
    :return: True - если точка лежит внутри отрезка, False - если точка не лежит внутри отрезка
    """
    max_x, min_x = max(p1[0], p2[0]), min(p1[0], p2[0])
    max_y, min_y = max(p1[1], p2[1]), min(p1[1], p2[1])
    return min_x <= p[0] <= max_x and min_y <= p[1] <= max_y

def inLine(p, p1, p2, eps=0.000001):
    """
    Функция принадлежности точки прямой линии
    :param p: Точка, которая проверяется на принадлежность
    :param p1: Первая точка, образующая линию
    :param p2: Вторая точка, образующая линию
    :param eps: Область сходимости
    :return: True - если точка принадлежит прямой линии, False - если не принадлежит линии
    """
    return (p2[1]-p1[1])*(p[0]-p1[0])-(p2[0]-p1[0])*(p[1]-p1[1]) < eps

def findPointsIntersectionEllipseWithLine(p1, p2, aE, bE, p0, alpha):
    """
    Фунция поиска точек пересечения эллипса с прямой линией
    :param p1: точка принадлежащая прямой L
    :param p2: точка принадлежащая прямой L (p1 != p2)
    :param a: длина главной полуоси эллипса
    :param b: длина второй полуоси эллипса
    :param p0: центр пересечения полуосей эллипса
    :param alpha: угол поворота эллипса в градусах
    :return: возвращает список координат точек пересечения, если пересечения нет, то возвращает пустой список
    """
    alpha_rad = np.pi * alpha / 180
    # Вычисление коэффициентов прямой
    p1_c, p2_c = subVec(p1, p0), subVec(p2, p0)
    p1_r, p2_r = rot(p1_c, -alpha_rad), rot(p2_c, -alpha_rad)
    kx, ky = subVec(p2_r, p1_r)
    x, y = p1_r
    # Рассчет коэффициентов квадратного уравнения
    a = bE * bE * kx * kx + ky * ky * aE * aE
    b = 2 * (x * kx * bE * bE + y * ky * aE * aE)
    c = aE * aE * (y * y - bE * bE) + bE * bE * x * x
    D = b*b - 4*a*c
    lst = []
    if D < -0.0000000001:
        return lst
    if D < 0.0000000001:
        t1 = -b / (2 * a)
        x1 = kx * t1 + x
        y1 = ky * t1 + y
        lst.append([x1, y1])
    else:
        t1 = (-b - np.sqrt(D))/ (2 * a)
        x1 = kx * t1 + x
        y1 = ky * t1 + y
        lst.append([x1, y1])
        t2 = (-b + np.sqrt(D)) / (2 * a)
        x2 = kx * t2 + x
        y2 = ky * t2 + y
        lst.append([x2, y2])

    for i in range(len(lst)):
        lst[i] = rot(lst[i], alpha_rad)
        lst[i] = addVec(lst[i], p0)

    return lst


def findPointsIntersectionEllipseWithSegment(p1, p2, aE, bE, p0, alpha):
    """
    Функция поиска точек пересечения эллипса с отрезком
    :param p1: точка принадлежащая прямой L
    :param p2: точка принадлежащая прямой L (p1 != p2)
    :param aE: длина главной полуоси эллипса
    :param bE: длина побочной полуоси эллипса
    :param p0: центр эллипса
    :param alpha: угол поворота эллипса в градусах
    :return: возвращает список координат точек пересечения, если пересечения нет, то возвращает пустой список
    """
    pts = findPointsIntersectionEllipseWithLine(p1, p2, aE, bE, p0, alpha)
    res = []
    if len(pts) > 0:
        for p in pts:
            if inSegmentLine(p, p1, p2):
                res.append(p)
    return res


def drawEllipse(screen, a, b, p0, alpha, color, w=1, N=50, fill = True):
    """
    Функция отрисовки эллипса
    :param screen: окно отрисовки
    :param a: длина главной полуось эллипса
    :param b: длина побочной полуось эллипса
    :param p0: центр эллипса
    :param alpha: угол поворота эллипса в градусах
    :param color: цвет отрисовки эллипса
    :param w: толщина линии эллипса, если w=0, то рисует сплощным цветом
    :param N: число линий отрисовки эллипса, рекомендуется N > 20
    :param fill: необязательный параметр, который определяется способ рисования эллипса
    если fill=True, то отрисовка с помощью pygame.draw.polygon,
    если fill=False, то отрисовка с помощью pygame.draw.line, используемая N раз
    """
    k = np.pi * 2 / N
    alpha_rad = np.pi*alpha/180
    def pt(t):
        stime, ctime = np.sin(t), np.cos(t)
        sa, ca = np.sin(alpha_rad), np.cos(alpha_rad)
        x = a * ctime
        y = b * stime
        x1 = x*ca - y*sa
        y1 = x*sa + y*ca
        return (x1  + p0[0], y1 + p0[1])

    if fill:
        pts = [pt(k*i) for i in range(N)]
        pg.draw.polygon(screen, color, pts, w)
    else:
        for i in range(N):
            pg.draw.line(screen, color, pt(k * (i - 1)), pt(k * i), w)

def drawRoute(screen, p0, pts, route, color=(0,200,200), colorp0=(255,200,0)):
    """
    Функция для отрисовки маршрута движения робота
    Рисуется линия маршрута

    :param screen: окно, на котором рисуется маршрут
    :param p0: положение робота
    :param pts: положение пунктов на окне приложения в порядке возрастания их номеров
    :param route: порядок обхода пунктов в соответствии с точками pts
    :param color: цвет маршрута
    :param colorp0: цвет отрисовки точки положения робота
    """
    pg.draw.circle(screen, colorp0, p0, 5)
    pg.draw.line(screen, color, p0, pts[route[0]], 2)
    pg.draw.circle(screen, color, pts[route[0]], 5)
    for i in range(1, len(route)):
        p1,p2=pts[route[i-1] ], pts[route[i]]
        pg.draw.line(screen, color, p1, p2, 2)
        pg.draw.circle(screen, color, pts[route[i]], 5)