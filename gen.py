import os
from rubiks import RubiksAction, RubiksState
import fire
import itertools
import random
from multiprocessing import Pool

DATA_DIR = './data/'
MAX_SCRAMBLE_LENGTH = 3

def generate_states(n):
    """
    Returns a list of random states.
    """
    states = []
    for i in range(n):
        state = RubiksState()
        k = random.randint(0, MAX_SCRAMBLE_LENGTH)
        for j in range(k):
            action = random.choice(state.get_next_actions())
            state = state.apply_action(action)
        states.append(state)
    return states


def entry(n_data=1,
          n_threads=1):
    """Generate the dataset and save it to disk."""

    # calculate how much to produce on each thread
    parts = []
    x = n_data
    while x > 0:
        parts.append(min(x, n_threads))
        x -= n_threads

    # generate the dataset into memory
    p = Pool(n_threads)
    states = p.map(generate_states, parts)
    states = list(itertools.chain(*states))  # flatten

    # save it onto disk
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    fname = os.path.join(DATA_DIR, 'dataset.txt')
    with open(fname, 'w') as f:
        f.write(str(len(states)))
        f.write('\n')
        for state in states:
            f.write(str(state))
            f.write('\n')
    print('Generated {} states and saved to {}'.format(len(states), fname))


if __name__ == '__main__':
    fire.Fire(entry)
