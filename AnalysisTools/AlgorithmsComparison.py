"""
FILE IS TO BE CLEANED AND REFACTORED

If you can read this warning, chances are the file will be unreadable and not useful.

"""
import sys
sys.path.append("..")

from Parameters import Configuration
import ipyparallel as ipp
import pickle
import numpy as np
from glob import glob
import re
import os

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

def load_NNSGA_agents(path):
    ags = list()
    for p in os.listdir(path):
        folder = os.path.join(path, p)
        if not os.path.isdir(folder):
            continue
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
        for ag in NNSGA_resume[2][-2:]:
            ags.append(ag)
    return ags


def load_POET_agents(path):
    ags = list()
    for p in os.listdir(path):
        folder = os.path.join(path, p)
        if not os.path.isdir(folder):
            continue
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
        for ea_pair in POET_resume:
            E, theta = ea_pair
            ags.append(theta)
    return ags


folder = ""
file_selected = False
while not file_selected:
    folder = input("Path to NNSGA executions folder : ")
    if not os.path.exists(folder):
        continue
    file_selected = True

NNSGA_ags = load_NNSGA_agents(folder)


folder = ""
file_selected = False
while not file_selected:
    folder = input("Path to POET executions folder : ")
    if not os.path.exists(folder):
        continue
    file_selected = True
POET_ags = load_POET_agents(folder)

# Generating environments ----------------------------------------------------------------------------------------------

print(f"Generating {nb_envs} new environments ...")

test_envs = list()
for i in range(nb_envs):
    test = Configuration.envFactory.new()
    nb_mut = np.random.randint(3, 15)
    for k in range(nb_mut):
        test = test.get_child()
    test_envs.append(test)


with open("../temp/Test_Environments.pickle", "wb") as f:
    pickle.dump(test_envs, f)
with open("../temp/POET_ag.pickle", "wb") as f:
    pickle.dump(POET_ags, f)
with open("../temp/NNSGA_ag.pickle", "wb") as f:
    pickle.dump(NNSGA_ags, f)
print("Saved tests environments and all agents as pickled files.")

cross_res_POET = [list() for i in range(nb_envs)]
cross_res_NNSGA = [list() for i in range(nb_envs)]

print("Evaluation (may take a while) ...")
for i in range(nb_envs):
    cross_res_POET[i] = Configuration.lview.map(test_envs[i], POET_ags)
    cross_res_NNSGA[i] = Configuration.lview.map(test_envs[i], NNSGA_ags)

cross_res_POET = np.array(cross_res_POET)
cross_res_NNSGA = np.array(cross_res_NNSGA)

np.save("cross_res_NNSGA", cross_res_NNSGA)
np.save("cross_res_POET", cross_res_POET)
