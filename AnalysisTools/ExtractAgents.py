import pickle
import numpy as np
from glob import glob
import re
import os


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

path = input("Save to :")

with open(path, "wb") as f:
    pickle.dump(NNSGA_ags, f)
print(f"Saved NNSGA agents to {path}")

folder = ""
file_selected = False
while not file_selected:
    folder = input("Path to POET executions folder : ")
    if not os.path.exists(folder):
        continue
    file_selected = True
POET_ags = load_POET_agents(folder)

path = input("Save to :")

with open(path, "wb") as f:
    pickle.dump(POET_ags, f)
print(f"Saved POET agents to {path}")