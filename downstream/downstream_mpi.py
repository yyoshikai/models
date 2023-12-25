"""
231120 作成
231204 3dvae以外のプロジェクトにも一般化

"""
import sys, os
import math
import argparse
from addict import Dict
sys.path.append(os.environ.get('TOOLS_DIR', "/workspace"))
from mpi4py import MPI
from tools.args import load_config2, load_config3
from tools.logger import default_logger
from models.downstream.downstream03 import main as main03


mains = {3: main03}

def main():

    config = load_config2("", [])
    logger = default_logger()
    version = config.get('version', 3)
    """
    if 'config_dir' not in config: # legacy
        logger.warning("config_dir is not specified. config of 3dvae is used.")
    """
    config_dir = config['config_dir']
    main_func = mains[version]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_process = len(config.args)
    if n_process % size != 0:
        logger.warning(f"Number of process({n_process}) % mpi size({size}) != 0. Redundant job exists.")
    if n_process > size:
        logger.info(f"Number of process({n_process}) > mpi size({size}). One job conducts up to ({math.ceil(n_process/size)}) process.")

    for i_process in range(rank, n_process, size):
        
        aconfig = Dict()
        for aconfig0 in config.args[:i_process+1]:
            aconfig.update(aconfig0)

        args = []
        for key, value in aconfig.items():
            args.append('--'+key)
            if isinstance(value, dict):
                args += list(value.values())
            else:
                args.append(value)
        pconfig = load_config3(args, config_dir, [])

        if 'variables' in pconfig:
            del pconfig['variables']
        main_func(pconfig, **pconfig)

main()