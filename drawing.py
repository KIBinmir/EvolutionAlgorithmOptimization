import pygame as pg
import numpy as np

pg.font.init()
font = pg.font.SysFont('Comic Sans MS', 20)
def drawText(screen, s, x, y):
    surf=font.render(s, True, (0,0,0))
    screen.blit(surf, (x,y))

def limAng(ang):
    while ang > np.pi: ang -= 2 * np.pi
    while ang <= -np.pi: ang += 2 * np.pi
    return ang

def rot(v, ang): #функция для поворота на угол
    s, c = np.sin(ang), np.cos(ang)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]

def addVec(v, v1):
    return [v[0] + v1[0], v[1] + v1[1]]

def subVec(v, v1):
    return [v[0] - v1[0], v[1] - v1[1]]

def inEllipse(p, a, b, p0, alpha):
    alpha_rad = np.pi * alpha / 180
    x1, y1 = p[0] - p0[0], p[1] - p0[1]
    x2, y2 = rot([x1, y1], -alpha_rad)
    return x2*x2/a/a + y2*y2/b/b <= 1.0

def inSegmentLine(p, p1, p2):
    max_x, min_x = max(p1[0], p2[0]), min(p1[0], p2[0])
    max_y, min_y = max(p1[1], p2[1]), min(p1[1], p2[1])
    return min_x <= p[0] <= max_x and min_y <= p[1] <= max_y

def calcDiskr(a, b, c):
    return b*b - 4*a*c

def findPointsIntersectionEllipseWithLine(p1, p2, aE, bE, p0, alpha):
    """
    :param p1: точка принадлежащая прямой L
    :param p2: точка принадлежащая прямой L (p1 != p2)
    :param a: длина главной полуоси эллписа
    :param b: длина второй полуоси эллипса
    :param p0: центр пересечения получосей эллипса
    :param alpha: угол поворота эллипса
    :return: возваращает список координаты точек, если пересечения нет, то возвращает пустой список
    """
    alpha_rad = np.pi * alpha / 180
    # Вычисление коэффициентов прямой
    p1_c, p2_c = subVec(p1, p0), subVec(p2, p0)
    p1_r, p2_r = rot(p1_c, -alpha_rad), rot(p2_c, -alpha_rad)
    kL = (p2_r[1] - p1_r[1]) / (p2_r[0] - p1_r[0])
    bL = p1_r[1] - kL*p1_r[0]
    # Рассчет коэффициентов квадратного уравнения
    a = bE * bE + kL * kL * aE * aE
    b = 2 * aE * aE * kL * bL
    c = aE * aE * (bL * bL - bE * bE)
    D = calcDiskr(a, b, c)
    lst = []
    if D < -0.0000000001:
        return lst
    if D < 0.0000000001:
        x1 = -b / (2 * a)
        y1 = kL*x1 + bL
        lst.append([x1, y1])
    else:
        x1 = (-b - np.sqrt(D))/ (2 * a)
        y1 = kL * x1 + bL
        lst.append([x1, y1])
        x2 = (-b + np.sqrt(D))/ (2*a)
        y2 = kL * x2 + bL
        lst.append([x2, y2])

    for i in range(len(lst)):
        lst[i] = rot(lst[i], alpha_rad)
        lst[i] = addVec(lst[i], p0)

    return lst

def findPointsIntersectionEllipseWithLine2(p1, p2, aE, bE, p0, alpha):
    """
    :param p1: точка принадлежащая прямой L
    :param p2: точка принадлежащая прямой L (p1 != p2)
    :param a: длина главной полуоси эллписа
    :param b: длина второй полуоси эллипса
    :param p0: центр пересечения получосей эллипса
    :param alpha: угол поворота эллипса
    :return: возваращает список координаты точек, если пересечения нет, то возвращает пустой список
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
    D = calcDiskr(a, b, c)
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
    pts = findPointsIntersectionEllipseWithLine2(p1, p2, aE, bE, p0, alpha)
    res = []
    if len(pts) > 0:
        for p in pts:
            if inSegmentLine(p, p1, p2):
                res.append(p)
    return res


def drawEllipse(screen, a, b, p0, alpha, color, w=1, N=50, fill = True):
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
