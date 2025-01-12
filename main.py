import pygame as pg
import numpy as np
import sys
from drawing import *
from evolution_comminvoyadger import *


def main():
    sz = (1200, 900)
    screen = pg.display.set_mode(sz)
    timer = pg.time.Clock()
    fps = 20

    numPoints = 50
    numAreas = 0
    field = Field(800, 600, 200, 150)
    field.initRobot()
    field.initPoints(numPoints)
    field.initDifficultAreas(numAreas)
    # route=list(range(0, len(pts)))

    distMatrix = getDistMatrix(field.points)
    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)

    routeBest, timeBest = None, None
    dst = 0
    ftnss = 0

    ga = TSPCEA(200, numPoints, numEliteIndivids=10)
    ga.init()
    ga.caclFitnessPopulation2(p0Times, timeMatrix)

    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                sys.exit(0)
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_g:
                    for i in range(100):
                        # ga.epoch(field.robot.getPos(), field.points)
                        ga.epoch2(p0Times, timeMatrix)
                        routeBest = ga.listEliteIndivids[0].route
                        timeBest = getTimeRoute2(p0Times, routeBest, timeMatrix)
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
                    routeBest = None
                if ev.key == pg.K_1:
                    numAreas = 1
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_2:
                    numAreas = 2
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_3:
                    numAreas = 3
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_4:
                    numAreas = 4
                    field.initDifficultAreas(numAreas)
                    distMatrix = getDistMatrix(field.points)
                    p0Dists = getDistsFromPoint2Points(field.robot.getPos(), field.points)
                    timeMatrix = getTimeMatrix(field.points, field.difficultAreas)
                    p0Times = getTimesFromPoint2Points(field.robot.getPos(), field.points, field.difficultAreas)
                    routeBest = None
                if ev.key == pg.K_5:
                    numAreas = 5
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

        dt = 1 / fps

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
    main()