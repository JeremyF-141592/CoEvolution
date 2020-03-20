import numpy as np
import argparse
from Parameters import baseEnv, flatConfig, agentFactory


def EA_init(args):
    """ Initialize both Environments and individuals according to a strategy, specified in the args."""

    # Initializing Environments ----------------------------------------------------------------------------------------
    envs = []
    if args.E_init == "flat":
        for i in range(args.Pop_size):
            envs.append(baseEnv(flatConfig))
    else:
        raise(argparse.ArgumentTypeError(f"Unknown Environment Inititialization strategy : {args.E_init}"))

    # Initializing Agents ----------------------------------------------------------------------------------------------
    ags = []
    if args.Theta_init == "random":
        for i in range(args.Pop_size):
            ag = agentFactory.new()
            ag.randomize()
            ags.append(ag)
    else:
        raise(argparse.ArgumentTypeError(f"Unknown Agent Inititialization strategy : {args.Theta_init}"))

    ea_pairs = [(envs[i], ags[i]) for i in range(args.Pop_size)]
    return ea_pairs
