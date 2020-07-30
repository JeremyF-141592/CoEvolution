"""
FILE IS TO BE CLEANED AND REFACTORED

If you can read this warning, chances are the file will be unreadable and not useful.

"""
import sys
sys.path.append("..")

from Parameters import Configuration
from AnalysisTools.ExtractAgents import load_agents_last_iteration
import ipyparallel as ipp
import pickle
import numpy as np
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

# Generating environments ----------------------------------------------------------------------------------------------

print(f"Generating {nb_envs} new environments ...")

test_envs = list()
for i in range(nb_envs):
    test = Configuration.envFactory.new()
    nb_mut = np.random.randint(5, 15)
    for k in range(nb_mut):
        test = test.get_child()
    test_envs.append(test)

path = input("path to parent folder : ")
while not os.path.exists(path) or not os.path.isdir(path):
    path = input("path to parent folder : ")

dir_list = list()
res_dic = dict()
for f in os.listdir(path):
    if os.path.isdir(os.path.join(path, f)):
        dir_list.append(f)
        res_dic[f] = [list() for i in range(nb_envs)]

ags_list = list()

for directory in dir_list:
    ags = load_agents_last_iteration(os.path.join(path, directory))
    ags_list.append(ags)

print("Evaluation (may take a while) ...")
for i in range(nb_envs):
    print(f"\tEnvironment {i}")
    for j in range(len(dir_list)):
        res_dic[dir_list[j]][i] = Configuration.lview.map(test_envs[i], ags_list[j])

with open(f"{path}/Results.pickle", "wb") as f:
    pickle.dump(res_dic, f)

print("Done.")
