"""
LatentTransformerのfeat_evalとgenerationを統合

"""
import sys, os
sys.path.append()
from mpi4py import MPI
from collections import OrderedDict
from tools.args import load_config2, load_config3
from evaluate import main as main_process

def main():
    config = load_config2("", [])
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if len(config.processes) != size:
        raise ValueError(f"# of process({len(config.processes)}) != mpi proc size({size})")

    arg_config = OrderedDict()
    for arg_config in config.processes[:rank+1]:
        arg_config.update(arg_config)

    args = []
    for key, value in arg_config.items():
        args.append(f"--{key}")
        if isinstance(value, dict):
            args += list(value)   
        elif isinstance(value, list):
            args += value
        else:
            args.append(value)
    process_config = load_config3(args, "", [])
    main_process(process_config)