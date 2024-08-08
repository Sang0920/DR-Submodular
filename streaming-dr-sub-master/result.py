import pandas as pd

def to_result(epsilon, bound, k, n_oracles, memory, n, time,
              influence, sum_x, chosen):
    return {'epsilon': epsilon,
            'bound': bound,
            'k': k,
            'n_oracles': n_oracles,
            'memory': memory,
            'n': n,
            'time': time,
            'influence': influence,
            'sum_x': sum_x,
            'chosen': chosen
            }
        
def to_pandas(results):
    tmp = dict()
    for result in results:
        for key in result:
            if key not in tmp:
                tmp[key] = list()
            tmp[key].append(result[key])
    return pd.DataFrame(tmp)

def save_result(df, file_name, mode='w'):
    df.to_csv(file_name, index=False, mode=mode)      

