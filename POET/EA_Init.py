import argparse
from Parameters import Configuration


def ea_init(args):
    """ Initialize both Environments and individuals according to a strategy, specified in the args."""

    # Initializing Environments ----------------------------------------------------------------------------------------
    envs = []
    if args.e_init == "flat":
        for i in range(args.pop_size):
            envs.append(Configuration.baseEnv(Configuration.envInit))
    else:
        raise(argparse.ArgumentTypeError(f"Unknown Environment Inititialization strategy : {args.e_init}"))

    # Initializing Agents ----------------------------------------------------------------------------------------------
    ags = []
    if args.theta_init == "random":
        for i in range(args.pop_size):
            ag = Configuration.agentFactory.new()
            ag.randomize()
            ags.append(ag)
    else:
        raise(argparse.ArgumentTypeError(f"Unknown Agent Inititialization strategy : {args.theta_init}"))

    ea_pairs = [(envs[i], ags[i]) for i in range(args.pop_size)]
    return ea_pairs
