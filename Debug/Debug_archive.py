from Parameters import Configuration
import matplotlib.pyplot as plt
import pickle
from glob import glob
import regex as re


Configuration.make()

# Resume execution -----------------------------------------------------------------------------------------------------

folder = "../POET_execution"

if folder != "":
    with open(f"{folder}/Archive.pickle", "rb") as f:
        archive = pickle.load(f)
    print(len(archive))
