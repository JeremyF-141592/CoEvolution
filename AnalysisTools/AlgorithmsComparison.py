#!/usr/bin/env python
"""
Generates N new environments, mutates them randomly and test multiple algorithm by loading their final states.

Specified folder should be 'parent_folder' such that final states of algorithm{1-2} are located in
'parent_folder/algorithm{1-2}/Iteration0.pickle'.

Results are saved as a dictionary under 'parent_folder/Results.pickle'.

"""
import sys
import os
import pickle
sys.path.append("..")

import ipyparallel as ipp
import numpy as np
from Parameters import Configuration
from AnalysisTools.ExtractAgents import load_agents_last_iteration

Configuration.make()

nb_envs = 100  # Amount of randomly generated Environments

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
    nb_mut = np.random.randint(5, 30)  # Mutates the environment 5 to 30 times
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
