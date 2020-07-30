import pickle
import numpy as np
from glob import glob
import re
import os


# Loading --------------------------------------------------------------------------------------------------------------

def load_agents_last_iteration(path):
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
            resume = pickle.load(f)
        print(f"Execution successfully loaded from {folder} .")
        if len(resume) == 3:
            if len(resume[2]) > 1:
                for ag in resume[2]:
                    ags.append(ag)
            else:
                for pop_ag in resume[0]:
                    for ag in pop_ag:
                        ags.append(ag)
        else:
            for ea_pair in resume:
                E, theta = ea_pair
                ags.append(theta)
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
