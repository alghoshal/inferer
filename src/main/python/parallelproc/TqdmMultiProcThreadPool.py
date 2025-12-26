from multiprocessing import Pool, Value
import os
from tqdm import tqdm
import time

os.nice(10)  # Be nice


def action(steps):
    for step in range(1, steps):
        y = step+1 # Nothing!
    return steps


def actWithProgBar():
    maxValues = 10**5
    procs = 5

    counter = 0
    with tqdm(total=maxValues, unit='B', unit_scale=True, desc="Status") as progBar:
        def update(steps):
            progBar.update(steps)

        with Pool(procs) as p:
            for steps in [(maxValues//procs)]*procs:
                p.apply_async(action, args=(steps,), callback=update)
                time.sleep(0.08)


if __name__ == '__main__':
    actWithProgBar()
