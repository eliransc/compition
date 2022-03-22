# from bayes_opt import BayesianOptimization
import time
from bayesian_optimization import BayesianOptimization
# Supress NaN warnings
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle as pkl

import os
import pandas as pd
import argparse
from __example__ import func1, get_curr_val

def func(x):
    return -3*x**2+6*x-3

# def func1(x1,x2):
#     return -((x1-3)**2+(x2-2)**2)

def main(args):




    # Bounded region of parameter space
    pbounds = {'a1': (0.0, 10),
               'a2': (0.0, 10),
               'a3': (0.0, 10),
               'a4': (0.0, 10),
               'a5': (0.0, 10)}


    optimizer = BayesianOptimization(
        f=get_curr_val,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum
        # is observed, verbose = 0 is silent
        random_state=1,
    )
    path = r'C:\Users\user\workspace\compition\compition\Business_Process_Optimization_Competition'
    path = os.path.join(path, 'prev_data_.pkl')
    start_time = time.time()
    if not os.path.exists(path):
        # df = pd.DataFrame([], columns=['x1', 'x2', 'target'])
        # x1 = [3, 2, 5, 2, 6, 1]
        # x2 = [4, 7, 8, 2, 8, 1]
        # target = [func1(x1[ind], x2[ind]) for ind in range(len(x1))]
        # df['x1'] = x1
        # df['x2'] = x2
        # df['target'] = target
        # pkl.dump(df, open(path, 'wb'))
        df = pd.DataFrame([], columns=['a1', 'a2', 'a3', 'a4', 'a5', 'target'])
        pkl.dump(df, open(path, 'wb'))

    optimizer.maximize(path, init_points=1, n_iter=6, )
    time_took = time.time() - start_time

    print(f"Total runtime: {str(time_took)}")
    print(optimizer.max)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='which settings are used', default='Gen_ph')
    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    main(args)

