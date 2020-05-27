"""
 -- Dataset Specification --
First line contains an integer N denoting the number of examples.
The next lines represent the N examples:
    the first line is an integer K denoting the heuristic,
    the second line is a list of numbers denoting sticker colors,
    the third line is an integer M representing the number of next states,
    the next M lines each contain a single next state:
        the next state is determined by another list of sticker colors.
"""

import os
from rubiks import RubiksAction, RubiksState
import fire
import itertools
import random
from multiprocessing import Pool

DATA_DIR = './data/'

def generate_states(n, max_scramble_length):
    """
    Returns a list of random states.
    """
    states = []
    for i in range(n):
        state = RubiksState()
        k = random.randint(0, max_scramble_length)
        for j in range(k):
            action = random.choice(state.get_next_actions())
            state = state.apply_action(action)
        states.append((k, state))
    return states


def entry(n_data=int(5e5),
          n_threads=4,
          name='dataset',
          dedup=False,
          max_scramble_length=5):
    """Generate the dataset and save it to disk."""

    # calculate how much to produce on each thread
    parts = []
    x = n_data
    while x > 0:
        parts.append(min(x, n_threads))
        x -= n_threads

    # generate the dataset into memory
    print('Generating data')
    p = Pool(n_threads)
    states = p.starmap(generate_states, zip(parts, [max_scramble_length]*len(parts)))
    states = list(itertools.chain(*states))  # flatten

    # dedup the values
    if dedup:
        uniq = set()
        deduped = []
        for k, state in states:
            if hash(state) not in uniq:
                uniq.add(hash(state))
                deduped.append((k, state))
        states = deduped

    # save it onto disk
    print('Saving to disk')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    fname = os.path.join(DATA_DIR, name+'.txt')
    with open(fname, 'w') as f:
        f.write(str(len(states)))
        f.write('\n')
        for k, state in states:
            f.write(str(k))
            f.write('\n')
            f.write(str(state))
            f.write('\n')
    print('Generated {} states and saved to {}'.format(len(states), fname))


if __name__ == '__main__':
    fire.Fire(entry)
