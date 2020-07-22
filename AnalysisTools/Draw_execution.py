"""
FILE IS TO BE CLEANED AND REFACTORED

If you can read this warning, chances are the file will be unreadable and not useful.

"""

from Parameters import Configuration
import matplotlib.pyplot as plt
import pickle
from glob import glob
import regex as re


Configuration.make()

save = False
pause_time = 1

# Resume execution -----------------------------------------------------------------------------------------------------

folder = "../old_executions/POET_execution"

if folder != "":
    filenames = glob(f"{folder}/*.pickle")[1:]
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    if not save:
        plt.show()
    for i in range(0, len(filenames), 25):
        plt.clf()
        ea_path = filenames[i]
        with open(f"{ea_path}", "rb") as f:
            ea_list_resume = pickle.load(f)
        plt.title(f"Iteration {i}")
        for k in range(len(ea_list_resume)):
            E, theta = ea_list_resume[k]
            plt.plot(E.terrain_y, label=f"{k}")
            E(theta, render=True)
        plt.legend()
        if save:
            plt.savefig(f"Iteration {i}.png")
        else:
            plt.pause(pause_time)
        #
        # for k in range(3, 8):
        #     E, theta = ea_list_resume[k]
        #     E(theta, render=True)
