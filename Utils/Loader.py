from Parameters import Configuration
from glob import glob
import os
import shutil
import json
import pickle
import re
from datetime import datetime
import sys


def resume_from_folder(folder, args):
    """Resume execution by loading archive, budget spent and the last iteration file generated."""
    with open(folder + "/HyperParameters.json", 'r') as f:
        args.__dict__ = json.load(f)
    filenames = glob(f"{folder}/*.pickle")
    filenames = list(filter(lambda x: "Iteration" in x, filenames))
    filenames.sort(key=lambda k: int(re.sub('\D', '', k)))
    # assume we only have relevant files in the folder, take the last sorted .pickle file
    ea_path = filenames[-1]
    numbers = ''.join((ch if ch in '0123456789' else ' ') for ch in ea_path)
    resume_from = int(numbers.split()[-1]) + 1
    if not os.path.exists(f"{ea_path}"):
        return OSError(f"No file was found at : {ea_path}")
    with open(f"{ea_path}", "rb") as f:
        iteration_resume = pickle.load(f)
    if os.path.exists(f"{folder}/Archive.pickle"):
        with open(f"{folder}/Archive.pickle", "rb") as f:
            Configuration.archive = pickle.load(f)
    else:
        print(f"\t No archive was found at : {folder}/Archive.pickle")

    if os.path.exists(f"{folder}/TotalBudget.json"):
        with open(f"{args.save_to}/TotalBudget.json", 'r') as f:
            budget_dic = json.load(f)
            Configuration.budget_spent = budget_dic["Budget_per_step"]
    else:
        print(f"\t No budget log was found at : {folder}/TotalBudget.json")
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
            fill_notice(args)
            print(f"{args.save_to} Successfully erased.")

    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
        fill_notice(args)

    with open(f"{args.save_to}/HyperParameters.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def fill_notice(args):
    with open(f"{os.path.dirname(__file__)}/Notice.txt", "r") as f:
        notice = f.read()

    info = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            os.path.basename(sys.argv[0]),
            Configuration.metric.__name__,
            type(Configuration.optimizer).__name__,
            Configuration.optimizer.__dict__,
            type(Configuration.agentFactory).__name__,
            Configuration.agentFactory.__dict__,
            type(Configuration.envFactory).__name__,
            Configuration.envFactory.__dict__,)

    with open(f"{args.save_to}/Notice.txt", "w") as f:
        f.write(notice.format(*info))
