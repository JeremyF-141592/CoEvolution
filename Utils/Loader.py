from Parameters import Configuration
from glob import glob
import os
import shutil
import json
import pickle
import re
import sys


def resume_from_folder(folder, args):
    """Resume execution by loading archive, budget spent and the last iteration file generated."""
    with open(folder + "/commandline_args.txt", 'r') as f:
        args.__dict__ = json.load(f)
    filenames = glob(f"{folder}/*.pickle")
    filenames = list(filter(lambda x: "Iteration" in x, filenames))
    filenames.sort(key=lambda k: int(re.sub('\D', '', k)))
    # assume we only have relevant files in the folder, take the last sorted .pickle file
    ea_path = filenames[-1]
    numbers = ''.join((ch if ch in '0123456789' else ' ') for ch in ea_path)
    resume_from = int(numbers.split()[-1])
    with open(f"{ea_path}", "rb") as f:
        iteration_resume = pickle.load(f)
    with open(f"{folder}/Archive.pickle", "rb") as f:
        Configuration.archive = pickle.load(f)
    with open(f"{args.save_to}/TotalBudget.json", 'r') as f:
        budget_dic = json.load(f)
        Configuration.budget_spent = budget_dic["Budget_per_step"]
    print(f"Execution successfully resumed from {folder} .")
    return iteration_resume, resume_from


def rm_folder_content(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def prepare_folder(args):
    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
    # Check if the folder to save to is empty, propose to abort otherwise
    if os.path.isdir(args.save_to) and len(os.listdir(args.save_to)) > 0:
        erase = ""
        while erase != "Y" and erase != "N":
            erase = input(f"\nWARNING : {args.save_to} is not empty, do you want to erase it ? (Y/N) : ")
            erase = erase.upper()
        if erase == "N":
            print("\n Please use the --save_to argument to specify a different folder.\n")
            sys.exit()
        else:
            rm_folder_content(args.save_to)
            print(f"{args.save_to} Successfully erased.")
