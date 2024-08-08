import json
import os 

with open('run.json', 'r') as file:
    settings = json.load(file)["settings"]
    tmux_session = (
            lambda alg, ds, k, b: 
            f"tmux new-session -d -s {alg}_{ds}_k{k}_b{b} 'python3 run_params.py {alg} {ds} {k} {b}'"
            )
    for setting in settings:
        os.system(
                tmux_session(
                    setting['alg'], 
                    setting['dataset'],
                    setting['k'],
                    setting['b']
            ))
