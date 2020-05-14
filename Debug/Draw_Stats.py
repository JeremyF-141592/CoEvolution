import matplotlib.pyplot as plt
from Utils.Stats import unpack_stats, mean_std
import os

file_selected = False
stat = dict()
path = ""
bonus = None

print("Type exit to exit, + after your key choice for a mean+std plot. \n")
while True:
    if not file_selected:
        path = input("Path to stat file : ")
        if path == "exit":
            break
        if not os.path.exists(path):
            continue
        stat = unpack_stats(path)
        file_selected = True

    print("0 : change file", end="")
    keys = [*stat.keys()]
    for k in range(len(keys)):
        print(f" - {k+1} : {keys[k]}", end="")
    choice = input("\n -> ")
    if len(choice.split()) == 2:
        choice, bonus = choice.split()

    try:
        choice = int(choice)
    except ValueError:
        print("Invalid input :", choice)
        continue
    if choice > len(keys) or choice < 0:
        print("Invalid input :", choice)
        continue

    if choice == 0:
        file_selected = False
        continue
    else:
        count = 0
        for tup in stat[keys[choice-1]]:
            plt.plot(tup[0], tup[1], label=count)
            count += 1
        plt.legend()
        plt.title(keys[choice-1])
        plt.show()

        if bonus is not None:
            if bonus == "+":
                x, m, s = mean_std(path, keys[choice-1])
                plt.fill_between(x, m+s, m-s, color=(0, 0.5, 1, 0.5))
                plt.plot(x, m, "r")
                plt.title(keys[choice-1])
                plt.show()
