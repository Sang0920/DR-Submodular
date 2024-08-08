import json
import subprocess
import os

import numpy as np

from algorithms import Algorithm2, Algorithm4, SieveStreaming, SomaCardinality, ThresholdGreedy

class Configuration:
    def __init__(self):
        self.algs = {
                'alg2': Algorithm2,
                'alg4': Algorithm4,
                'tg': ThresholdGreedy,
                'sieve': SieveStreaming,
                'soma': SomaCardinality
                }
        self.data_dir = 'data'
        self.log_dir = 'log'
        with open('dataset_meta.json', 'r') as file:
            self.datasets = json.load(file)
        self.output_dir = 'output'
        self.k_values = np.arange(100, 301, 50)     
        self.b_max = 3
        self.epsilon = 0.1
        self.init_dir()

    def get_alg_ids(self):
        return list(self.algs.keys())

    def mkdir(self, dir):
        subprocess.run(f'mkdir {dir}', shell=True)

    # def init_dir(self):
    #     result = subprocess.check_output('ls', shell=True, universal_newlines=True)
    #     result = result.split('\n')
    #     if self.data_dir not in result:
    #         self.mkdir(self.data_dir) 
    #     if self.log_dir not in result:
    #         self.mkdir(self.log_dir)
    #     if self.output_dir not in result:
    #         self.mkdir(self.output_dir)
        

    def init_dir(self):
        directories = [self.data_dir, self.log_dir, self.output_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)