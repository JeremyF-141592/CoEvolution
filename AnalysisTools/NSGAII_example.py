#!/usr/bin/env python
"""
    2D Animation of the NSGA2 algorithm.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from Algorithms.NSGA2.NSGAII_core import *

iterations = 30
delay = 0.01
population_size = 10
generation_size = 10

save = False
pause = 1


def new_point():
    return np.random.random(), np.random.random()


def prepare_plot(phase, pause):
    plt.pause(pause)
    plt.clf()
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.title("Iteration {} : {:>25}".format(t, phase))


colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]

if not os.path.exists("../NSGA_anim") and save:
    os.mkdir("../NSGA_anim")
plt.show()
population = [new_point() for i in range(population_size)]  # list instead of array, flexible structure
for t in range(iterations):

    # Generate new individuals -----------------------------------------------------------------------------------------
    new_pop = [new_point() for i in range(generation_size)]
    full_pop = population + new_pop

    prepare_plot("Generation", delay)
    for point in full_pop:
        plt.plot(point[0], point[1], "o", color="blue")

    if save:
        plt.savefig(f"../NSGA_anim/{t}-0")
    else:
        plt.pause(pause)

    # Non dominated sort -----------------------------------------------------------------------------------------------
    ranks = np.array(fast_non_dominated_sort(full_pop))   # get front index for each individual

    prepare_plot("Non-dominated sorting", delay)

    fronts = [list() for i in range(ranks.max() + 1)]  # store fronts as lists
    for i in range(len(full_pop)):
        point = full_pop[i]
        fronts[ranks[i]].append(point)
        plt.plot(point[0], point[1], "o", color=colors[ranks[i] % len(colors)])

    for j in range(len(fronts)):
        fronts[j].sort(key=lambda x: x[0])
        for i in range(len(fronts[j])-1):
            plt.plot([fronts[j][i][0], fronts[j][i+1][0]], [fronts[j][i][1], fronts[j][i+1][1]], color=colors[j % len(colors)])

    if save:
        plt.savefig(f"../NSGA_anim/{t}-1")
    else:
        plt.pause(pause)

    population = list()  # New population
    last_front = 0
    for i in range(len(fronts)):
        if len(population) + len(fronts[i]) > population_size:
            break
        population = population + fronts[i]  # add as much fronts as possible in the new population
        last_front = i + 1

    # Crowding distance  -----------------------------------------------------------------------------------------------

    prepare_plot("Crowding distance", delay)
    for point in population:
        plt.plot(point[0], point[1], "o", color="red")

    if last_front < len(fronts):
        cdistance = np.array(crowding_distance(fronts[last_front]))
        cdist_sort = cdistance.argsort()[::-1]  # get (position in the list) of the points maximizing crowding distance

        # fill the population with less crowded individuals of the last front
        for i in range(population_size - len(population)):
            less_crowded_point = fronts[last_front][cdist_sort[i]]
            population.append(less_crowded_point)
            plt.plot(less_crowded_point[0], less_crowded_point[1], "o", color="blue")

    if save:
        plt.savefig(f"../NSGA_anim/{t}-2")
    else:
        plt.pause(pause)
