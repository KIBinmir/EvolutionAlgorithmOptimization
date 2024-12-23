import pygame as pg
import numpy as np
import sys
from drawing import *


def test_intersection():
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
        drawEllipse(screen, a, b, p0, alpha, (0,0,40, 128), w=0, fill=True)
        drawText(screen, f"Point in ellipse = {state}", 5, 5)
        drawText(screen, f"Point = {p}", 5, 25)
        drawText(screen, f"Angle = {alpha}", 5, 45)
        pg.draw.circle(screen, (255,0,0), p, 3)

        pg.display.flip()
        timer.tick(fps)


if __name__ == '__main__':
    test_intersection()
    #test_ellipse()
    #test()