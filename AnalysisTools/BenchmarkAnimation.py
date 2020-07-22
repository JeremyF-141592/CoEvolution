"""
FILE IS TO BE CLEANED AND REFACTORED

If you can read this warning, chances are the file will be unreadable and not useful.

"""

from Utils.Stats import unpack_stats
from Objects.Environments.Benchmark import *


Configuration.make()

bench = diag_gaussian

t_local = 15
t_global = 15

archive = list()
full_ev2 = list()
env_ogs = None
stats = unpack_stats("../NewStats.json")

best_colors = ["green", "yellow", "cyan"]

# plt.show()
for iteration in range(0, 300):
    local = iteration % (t_local + t_global) < t_local

    plt.clf()
    plt.title("Iteration {} {:>5}".format("locale" if local else "globale", iteration))
    print(iteration)

    best_args = [list(), list()]
    for i in range(2):
        for k in range(len(stats[f'Objective_{i}-argmax'])):
            if iteration in stats[f'Objective_{i}-argmax'][k][0]:
                ajusted = iteration - stats[f'Objective_{i}-argmax'][k][0][0]
                best_args[i].append(stats[f'Objective_{i}-argmax'][k][1][ajusted])

    for i in range(2):
        for k in range(len(stats[f'Objective_general-arg{i}'])):
            if iteration in stats[f'Objective_general-arg{i}'][k][0]:
                ajusted = iteration - stats[f'Objective_general-arg{i}'][k][0][0]
                best_args[i].append(stats[f'Objective_general-arg{i}'][k][1][ajusted])

    env_ogs = list()
    for k in range(len(stats['xy_benchmark'])):
        if iteration in stats['xy_benchmark'][k][0]:
            ajusted = iteration - stats['xy_benchmark'][k][0][0]
            env_ogs.append(stats['xy_benchmark'][k][1][ajusted][1])

    ags = list()
    for k in range(len(stats['xy_benchmark'])):
        if iteration in stats['xy_benchmark'][k][0]:
            ajusted = iteration - stats['xy_benchmark'][k][0][0]
            ags.append(stats['xy_benchmark'][k][1][ajusted][0])

    cal = np.array(env_ogs)
    # full_ev += evs
    bounds = [-50, 50, -5, 25]

    size = 200
    k = np.linspace(bounds[0], bounds[1], num=size)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            a[j, i] = bench(k[i], k[j])

    plt.imshow(a, cmap='Greys', interpolation='nearest', extent=[bounds[0], bounds[1], bounds[2], bounds[3]])
    plt.xlim(bounds[0], bounds[1])
    plt.ylim(bounds[2], bounds[3])
    for env_og in env_ogs:
        plt.axhline(env_og, linestyle="dashed")

    ev_best_counter = 0
    for a in range(1, len(ags)):
        if env_ogs[a] == env_ogs[a-1]:
            plt.plot(ags[a], env_ogs[a], "or")
    for a in range(len(ags)):
        if a == 0 or env_ogs[a] != env_ogs[a-1]:
            ev_best_counter = min(ev_best_counter, len(best_args[0]) -1)
            print(len(best_args[0]), ev_best_counter)
            print(len(ags), best_args[0][ev_best_counter])
            plt.plot(ags[best_args[0][ev_best_counter]], env_ogs[a], "o", color=best_colors[0])
            plt.plot(ags[best_args[1][ev_best_counter]], env_ogs[a], "o", color=best_colors[1])
            ev_best_counter += 1
    plt.xlabel("Agent")
    plt.ylabel("Environment")
    plt.pause(0.01)
    # plt.savefig(f"../anim2/{iteration}.png")
