from Parameters import Configuration
import matplotlib.pyplot as plt
import pickle
from glob import glob
import regex as re


Configuration.make()

# Resume execution -----------------------------------------------------------------------------------------------------

folder = "../POET_execution"

if folder != "":
    filenames = glob(f"{folder}/*.pickle")[1:]
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

    # plt.show()
    for i in range(0, len(filenames), 50):
        plt.clf()
        ea_path = filenames[i]
        with open(f"{ea_path}", "rb") as f:
            ea_list_resume = pickle.load(f)
        plt.title(f"Iteration {i}")
        for k in range(8):
            E, theta = ea_list_resume[k]
            E.cppn.print()
            plt.plot(E.terrain_y, label=f"{k}")
        plt.legend()
        plt.savefig(f"Iteration {i}.png")
        # for k in range(3, 8):
        #     E, theta = ea_list_resume[k]
        #     E(theta, render=True)
