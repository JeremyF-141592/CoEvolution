"""
FILE IS TO BE CLEANED AND REFACTORED

If you can read this warning, chances are the file will be unreadable and not useful.

"""

from Parameters import Configuration
import numpy as np
from glob import glob
import pickle
import re


# ---------------------------------------------------------------------------

path = "../POET_execution"
nb_points = 50

# ---------------------------------------------------------------------------


Configuration.make()

filenames = glob(f"{path}/*.pickle")
filenames = list(filter(lambda x: "Iteration" in x, filenames))
filenames.sort(key=lambda k: int(re.sub('\D', '', k)))
# assume we only have relevant files in the folder, take the last sorted .pickle file
ea_path = filenames[-1]
resume_from = len(filenames)
with open(f"{ea_path}", "rb") as f:
    last_iteration = pickle.load(f)
print(f"Last iteration successfully loaded from {path}.")

new_envs = list()
dists = list()

res = list()

theta_list = list()
for ea_pair in last_iteration:
    E, theta = ea_pair
    theta_list.append(theta)

for i in range(nb_points):
    dists.append(list())
    choices = np.random.choice(np.arange(len(last_iteration)), size=2)

    env1 = last_iteration[choices[0]][0]
    env2 = last_iteration[choices[1]][0]
    new = env1.crossover(env2)
    for ea_pair in last_iteration:
        E, theta = ea_pair
        dists[-1].append(np.linalg.norm(np.array(E.terrain_y) - np.array(new.terrain_y)))

    fitness = Configuration.lview.map(new, theta_list)
    res.append(fitness)

with open("FitDist_fit.pickle", "wb") as f:
    pickle.dump(res, f)

with open("FitDist_dist.pickle", "wb") as f:
    pickle.dump(dists, f)
