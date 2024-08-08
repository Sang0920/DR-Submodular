from datetime import datetime
import time
import json
import sys

from loguru import logger
import numpy as np

from result import *
from configs import Configuration
from tools import read_dataset
from objective_functions import BudgetAllocation

def run(config, alg_arg, dataset, k):
    log_dir = (f"{config.log_dir}/" 
               + f"{alg_arg}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')}.log")
    # logger.add(log_dir, rotation="20 MB")
    # loop thru argument datasets
    if dataset not in config.datasets:
        logger.info(f'Error: no dataset called "{dataset}" in configuration')
        sys.exit(1)
    ds = config.datasets[dataset]
    dsdir = ds['dir']
    dsout = ds['output']
    results = []
    logger.info(f'Processing {dsdir}')
    E, targets, adj, w = read_dataset(dsdir, 
                                      max_weight=ds['max_weight'],
                                      delimiter=ds['delimiter'], 
                                      reverse=ds['reverse_st'])
    n = len(E)
    f = BudgetAllocation(E, adj, w)
    logger.info(f'### Running for b={config.b_max} and k={k} ###')
    f.reset() 
    B = np.full(n, config.b_max)
    start = time.time()
    alg = config.algs[alg_arg](e_arr=E, b_arr=B, f=f, k=k, 
                               epsilon=config.epsilon)
    x = alg.run()
    duration = time.time() - start
    results.append(to_result(config.epsilon, config.b_max, int(k), 
                             f.count, alg.memory, n,
                             duration, float(f(x)),
                             int(np.sum(x)),
                             len(x[x>0])
                             ))
    json_data = {
            'alg': alg_arg,
            'data': results[-1]
            }
    logger.info(f"""
    -------Result-------
    {json.dumps(json_data)}
    --------------------""")
    df = to_pandas(results)
    save_result(df, f'{config.output_dir}/{alg_arg}_k{k}_b{config.b_max}_{dsout}')


if __name__ == "__main__":
    config = Configuration()
    lenarg = len(sys.argv)
    if lenarg < 2:
        logger.info("Error: require algorithm's id "
                    + f"\n Available ids: {config.get_alg_ids()}")
        sys.exit(1)
    if sys.argv[1] not in config.get_alg_ids():
        logger.info(f'Error: "{sys.argv[1]}" is not a valid id'
                    + f"\nAvailable ids: {config.get_alg_ids()}")
        sys.exit(1)
    if lenarg < 3:
        logger.info('Error: require algorithm and at least one dataset argument')
        sys.exit(1)
    if lenarg < 4:
        logger.info("Error: require k's value")
        sys.exit(1)
    if lenarg == 5:
        config.b_max = int(sys.argv[4])
    run(config, sys.argv[1], sys.argv[2], int(sys.argv[3]))

