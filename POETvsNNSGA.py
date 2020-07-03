from Parameters import Configuration
from Utils.Stats import unpack_stats
import ipyparallel as ipp
import pickle
import json
import numpy as np
import os
from glob import glob
import re

Configuration.make()

nb_envs = 100

# Ipyparallel ----------------------------------------------------------------------------------------------------------
# Local parallelism, make sure that ipcluster is started beforehand otherwise this will raise an error.
Configuration.rc = ipp.Client()
with Configuration.rc[:].sync_imports():
    from Parameters import Configuration
Configuration.rc[:].execute("Configuration.make()")
Configuration.lview = Configuration.rc.load_balanced_view()
Configuration.lview.block = True

# Loading --------------------------------------------------------------------------------------------------------------

folder = ""
file_selected = False
while not file_selected:
    folder = input("Path to NNSGA execution folder : ")
    if not os.path.exists(folder):
        continue
    stat = unpack_stats(folder + "/Stats.json")
    file_selected = True


filenames = glob(f"{folder}/*.pickle")
filenames = list(filter(lambda x: "Iteration" in x, filenames))
filenames.sort(key=lambda k: int(re.sub('\D', '', k)))
# assume we only have relevant files in the folder, take the last sorted .pickle file
ea_path = filenames[-1]
numbers = ''.join((ch if ch in '0123456789' else ' ') for ch in ea_path)
resume_from = int(numbers.split()[-1])
with open(f"{ea_path}", "rb") as f:
    NNSGA_resume = pickle.load(f)
print(f"Execution successfully loaded from {folder} .")

folder = ""
file_selected = False
while not file_selected:
    folder = input("Path to POET execution folder : ")
    if not os.path.exists(folder):
        continue
    stat = unpack_stats(folder + "/Stats.json")
    file_selected = True


filenames = glob(f"{folder}/*.pickle")
filenames = list(filter(lambda x: "Iteration" in x, filenames))
filenames.sort(key=lambda k: int(re.sub('\D', '', k)))
# assume we only have relevant files in the folder, take the last sorted .pickle file
ea_path = filenames[-1]
numbers = ''.join((ch if ch in '0123456789' else ' ') for ch in ea_path)
resume_from = int(numbers.split()[-1])
with open(f"{ea_path}", "rb") as f:
    POET_resume = pickle.load(f)
with open(f"{folder}/Archive.pickle", "rb") as f:
    POET_archive = pickle.load(f)
print(f"Execution successfully loaded from {folder} .")

# NNSGA ----------------------------------------------------------------------------------------------------------------

NNSGA_ag = NNSGA_resume[0]
NNSGA_env = NNSGA_resume[1]
NNSGA_ag_general = NNSGA_resume[2]

# POET -----------------------------------------------------------------------------------------------------------------

POET_ag = list()
POET_env = list()
for ea_pair in POET_resume:
    E, theta = ea_pair
    POET_ag.append(theta)
    POET_env.append(E)

# Generating environments ----------------------------------------------------------------------------------------------

test_envs = list()
for i in range(nb_envs):
    test = Configuration.envFactory.new()
    nb_mut = np.random.randint(3, 15)
    for k in range(nb_mut):
        test = test.get_child()
    test_envs.append(test)

cross_res_POET = [list() for i in range(nb_envs)]
cross_res_NNSGA = [list() for i in range(nb_envs)]

for i in range(nb_envs):
    cross_res_POET[i] = Configuration.lview.map(test_envs[i], POET_ag)
    cross_res_NNSGA[i] = Configuration.lview.map(test_envs[i], NNSGA_ag)

cross_res_POET = np.array(cross_res_POET)
cross_res_NNSGA = np.array(cross_res_NNSGA)

np.save("cross_res_NNSGA", cross_res_NNSGA)
np.save("cross_res_POET", cross_res_POET)
