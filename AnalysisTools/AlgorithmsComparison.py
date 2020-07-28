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

path1 = input("Path to agents 1 :")
with open(path1, "rb") as f:
    ags1 = pickle.load(f)
print(f"\tSuccessfully loaded agents from {path1}")

path2 = input("Path to agents 1 :")
with open(path2, "rb") as f:
    ags2 = pickle.load(f)
print(f"\tSuccessfully loaded agents from {path2}")

with open("./Test_Environments.pickle", "wb") as f:
    pickle.dump(test_envs, f)

cross_res_1 = [list() for i in range(nb_envs)]
cross_res_2 = [list() for i in range(nb_envs)]

print("Evaluation (may take a while) ...")
for i in range(nb_envs):
    cross_res_1[i] = Configuration.lview.map(test_envs[i], ags1)
    cross_res_2[i] = Configuration.lview.map(test_envs[i], ags2)

cross_res_1 = np.array(cross_res_1)
cross_res_2 = np.array(cross_res_2)

np.save(f"Results_{path1}", cross_res_1)
np.save(f"Results_{path2}", cross_res_2)
