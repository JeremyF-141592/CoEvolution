import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
sys.path.insert(1, "../")

from Utils.Stats import unpack_stats, mean_std, min_max


def name_save_fig():
    title = input("Title : ")
    xlab = input("X label : ")
    ylab = input("Y label : ")
    out = input("Output file : ")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(out)
    print("Figure saved at :", out)


file_selected = False
stat = dict()
path = ""

folder_mode = False

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
    bonus = None
    if len(choice.split()) > 1:
        bonus = choice.split()
        choice = bonus[0]
        bonus = bonus[1:]

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
        maxi_x = 0
        for tup in stat[keys[choice-1]]:
            if keys[choice-1] == "x_benchmark":
                bidule0 = list()
                bidule1 = list()
                for k in range(1, len(tup[1])):
                    if abs(tup[1][k] - tup[1][k-1]) > 0.2:
                        bidule1.append(np.nan)
                        bidule0.append(np.nan)
                    bidule1.append(tup[1][k])
                    bidule0.append(tup[0][k])
                plt.plot(bidule0, bidule1, "b", label=count)
            else:
                plt.plot(tup[0], tup[1], label=count)
                maxi_x = max(maxi_x, tup[0][-1])
                plt.legend()
            count += 1

        if bonus is not None:
            if "||" in bonus:
                try:
                    vertical1 = int(bonus[bonus.index("||") + 1])
                    vertical2 = int(bonus[bonus.index("||") + 2])
                    for k in range(0, maxi_x, vertical2 + vertical1):
                        if k != 0 and vertical1 != 0:
                            plt.axvline(k, linestyle="--", color="#babbff")
                    for k in range(0, maxi_x, vertical2 + vertical1):
                        if vertical1 != 0:
                            plt.axvline(k + vertical1 -1, linestyle="--", color="#ffbaba")
                    blue_patch = mpatches.Patch(color='#babbff', label='Local iterations')
                    red_patch = mpatches.Patch(color='#ffbaba', label='Global iterations')

                    # handles, labels = plt.gca().get_legend_handles_labels()
                    handles = [blue_patch, red_patch]
                    plt.legend(handles=handles)
                finally:
                    pass

            if "save" in bonus and "+" not in bonus:
                name_save_fig()

        else:
            plt.title(keys[choice-1])
        plt.show()

        if bonus is not None:
            if "+" in bonus:
                x, m, s = mean_std(path, keys[choice-1])
                if "l" in bonus:
                    x = [x[i] for i in range(10, len(x)-10)]
                    m = [m[i-10:i+10].mean() for i in range(10, len(m)-10)]
                    m = np.array(m)
                    s = [s[i-10:i+10].mean() for i in range(10, len(s)-10)]
                    s = np.array(s)
                plt.fill_between(x, m+s, m-s, color=(0, 0.5, 1, 0.5))
                plt.plot(x, m, "r")
                plt.title(keys[choice-1])
                if "save" in bonus:
                    name_save_fig()
                plt.show()

            if "=" in bonus:
                x, mini, maxi = min_max(path, keys[choice-1])
                if "l" in bonus:
                    x = [x[i] for i in range(10, len(x)-10)]
                    mini = [mini[i-10:i+10].mean() for i in range(10, len(mini)-10)]
                    mini = np.array(mini)
                    maxi = [maxi[i-10:i+10].mean() for i in range(10, len(maxi)-10)]
                    maxi = np.array(maxi)
                plt.plot(x, mini, "b", label="Minimum")
                plt.plot(x, maxi, "r", label="Maximum")
                plt.title(keys[choice-1])
                plt.legend()
                if "save" in bonus:
                    name_save_fig()
                plt.show()
