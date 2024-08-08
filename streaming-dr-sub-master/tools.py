import os
import csv

from tqdm import tqdm
import psutil
import numpy as np


def get_memory():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def read_dataset(dataset, max_weight=5, 
                 delimiter='\t', reverse=False):
    """Return sources, targets, adjacencies and weights from target to source from dataset"""
    sources = set()
    targets = set()
    edges = list()
    num_lines = sum(1 for _ in open(dataset))
    with tqdm(total=num_lines, position=0,
              leave=False, desc="Reading data") as pbar:
        with open(dataset, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                try:
                    u = int(row[0])
                    v = int(row[1])
                    w = float(row[2]) / max_weight
                    if reverse:
                        sources.add(v)
                        targets.add(u)
                        edges.append([v,u,w])
                    else:
                        sources.add(u)
                        targets.add(v)
                        edges.append([u,v,w])
                except:
                    pass
                pbar.update(1)
    sources = list(sources)
    targets = list(targets)
    n = len(sources)
    m = len(targets)
    w = np.zeros((m, n))
    source_index = {s: index for index, s in enumerate(sources)}
    target_index = {t: index for index, t in enumerate(targets)}
    adjacency = [set() for _ in range(n)]
    with tqdm(total = len(edges),
              position=0, leave=False, 
              desc="Preparing data") as pbar:
        for edge in edges:
            u, v, weight = edge
            w[target_index[v], source_index[u]] = 1 - weight
            adjacency[source_index[u]].add(target_index[v])
            pbar.update(1)
        return list(range(n)), list(range(m)), adjacency, w

class OracleCounter:
    def __init__(self, f):
        self.count = 0
        self.f = f

    def __call__(self, *args, **kwds):
        self.count += 1
        return self.f(*args, **kwds)

    def reset(self):
        self.count = 0
