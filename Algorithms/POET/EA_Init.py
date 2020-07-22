import argparse
from Parameters import Configuration


def ea_init(args):
    ea_pairs = [(Configuration.envFactory.new(), Configuration.agentFactory.new()) for i in range(args.pop_size)]
    return ea_pairs
