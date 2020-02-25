from abstract import State, Action, Graph
import random
import os
import pickle
import time
import numpy as np
from multiprocessing import Pool
import fire
import torch

# "safe mode" can be enabled here if there's a bug or
# something. It enables several checks for correctness,
# but it also slows down the graph search.
SAFE_MODE = False

MODEL_CHECKPOINT_DIR = os.path.join(
        os.path.dirname(__file__), 
        'model_checkpoints/')

MAX_SCRAMBLE_LENGTH = 19

RUBIKS_PERMS = {
    'F': {
        10: 12, 12: 18, 18: 16, 16: 10,
        11: 15, 15: 17, 17: 13, 13: 11,
         7: 37, 37: 21, 21: 54, 54:  7,
         8: 40, 40: 20, 20: 51, 51:  8,
         9: 43, 43: 19, 19: 48, 48:  9
    },

    'R': {
        37: 39, 39: 45, 45: 43, 43: 37,
        38: 42, 42: 44, 44: 40, 40: 38,
        30: 21, 21: 12, 12:  3,  3: 30,
        33: 24, 24: 15, 15:  6,  6: 33,
        36: 27, 27: 18, 18:  9,  9: 36
    },

    'L': {
        46: 48, 48: 54, 54: 52, 52: 46,
        47: 51, 51: 53, 53: 49, 49: 47,
         1: 10, 10: 19, 19: 28, 28:  1,
         4: 13, 13: 22, 22: 31, 31:  4,
         7: 16, 16: 25, 25: 34, 34:  7
    },

    'D': {
        19: 21, 21: 27, 27: 25, 25: 19,
        20: 24, 24: 26, 26: 22, 22: 20,
        16: 43, 43: 30, 30: 52, 52: 16,
        17: 44, 44: 29, 29: 53, 53: 17,
        18: 45, 45: 28, 28: 54, 54: 18
    },

    'U': {
         1:  3,  3:  9,  9:  7,  7:  1, 
         2:  6,  6:  8,  8:  4,  4:  2, 
        37: 10, 10: 46, 46: 36, 36: 37,
        38: 11, 11: 47, 47: 35, 35: 38,
        39: 12, 12: 48, 48: 34, 34: 39
    },

    'B': {
        28: 30, 30: 36, 36: 34, 34: 28,
        29: 33, 33: 35, 35: 31, 31: 29,
        25: 45, 45:  3,  3: 46, 46: 25,
        26: 42, 42:  2,  2: 49, 49: 26,
        27: 39, 39:  1,  1: 52, 52: 27
    }
}

# The extended permutations are just compositions of 
# the basic permutations, but count as single moves.
EXTENDED_RUBIKS_PERMS = {
        'F*': (3, 'F'),
        'R*': (3, 'R'),
        'L*': (3, 'L'),
        'D*': (3, 'D'),
        'U*': (3, 'U'),
        'B*': (3, 'B'),
        'F2': (2, 'F'),
        'R2': (2, 'R'),
        'L2': (2, 'L'),
        'D2': (2, 'D'),
        'U2': (2, 'U'),
        'B2': (2, 'B')
}

RUBIKS_ACTIONS = [
    'F' , 'R' , 'L' , 'D' , 'U' , 'B' ,
    'F*', 'R*', 'L*', 'D*', 'U*', 'B*',
    'F2', 'R2', 'L2', 'D2', 'U2', 'B2'
]

RUBIKS_COLORS = {
     1: 1,   2: 1,   3: 1,   4: 1,   5: 1,   6: 1,   7: 1,   8: 1,   9: 1,
    10: 2,  11: 2,  12: 2,  13: 2,  14: 2,  15: 2,  16: 2,  17: 2,  18: 2,
    19: 3,  20: 3,  21: 3,  22: 3,  23: 3,  24: 3,  25: 3,  26: 3,  27: 3,
    28: 4,  29: 4,  30: 4,  31: 4,  32: 4,  33: 4,  34: 4,  35: 4,  36: 4,
    37: 5,  38: 5,  39: 5,  40: 5,  41: 5,  42: 5,  43: 5,  44: 5,  45: 5,
    46: 6,  47: 6,  48: 6,  49: 6,  50: 6,  51: 6,  52: 6,  53: 6,  54: 6
}


class RubiksAction(Action):

    def get_perm(self, name):

        if name in RUBIKS_PERMS:
            mult, basic_name = 1, name
        elif name in EXTENDED_RUBIKS_PERMS:
            mult, basic_name = EXTENDED_RUBIKS_PERMS[name]
        else:
            raise ValueError('RubiksAction.get_perm :: Invalid perm name {}'.format(name))

        # The above permutations are not written fully.
        # They only map the sticker spots that they change.
        # All other stickers are not represented.
        # So, we must add those invariant mappings as well.
        perm = dict()
        for i in range(1, 6*9+1):
            if i in RUBIKS_PERMS[basic_name]:
                perm[i] = RUBIKS_PERMS[basic_name][i]
            else:
                perm[i] = i
        return (mult, perm)

    def __init__(self, name):
        if SAFE_MODE:
            if name not in RUBIKS_ACTIONS:
                raise ValueError('RubiksAction.__init__ :: Invalid name {}'.format(name))
        self.name = name
        self.mult, self.perm = self.get_perm(name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class RubiksState(State):
    """
    This is simply a wrapper around a tuple, which holds
    the sticker numbers of a Rubik's cube state.
    """

    def __init__(self, perm: tuple = tuple(range(1, 6*9+1))):
        # defaults to the solved state
        if SAFE_MODE:
            if tuple(sorted(perm)) != tuple(range(1, 6*9+1)):
                raise ValueError('RubiksState.__init__ :: Invalid permutation {}'.format(perm))

        self.perm = perm

    def __hash__(self):
        return hash(self.perm)

    def __eq__(self, other):
        return self.perm == other.perm

    def __lt__(self, other):
        return self.perm[1] < other.perm[1]

    def __str__(self):
        return ' '.join(map(str, self.perm))

    def __repr__(self):
        return str(self.perm)

    def apply_action(self, a: RubiksAction) -> 'RubiksState':
        if SAFE_MODE:
            if type(a) != RubiksAction:
                raise ValueError('RubiksState.apply_action :: bad action type {}'.format(
                    type(a)))

        cur_perm = self.perm
        for _ in range(a.mult):
            new_perm = list(range(1, 6*9+1))
            for i in range(1, 6*9+1):
                new_perm[a.perm[i] - 1] = cur_perm[i - 1]
            cur_perm = new_perm
        return RubiksState(tuple(cur_perm))

    def get_next_actions(self) -> list:
        actions = [RubiksAction(a) for a in RUBIKS_ACTIONS]
        random.shuffle(actions)
        return actions

    def trainable(self):
        x = [RUBIKS_COLORS[i] for i in self.perm]
        x = np.array(x) / 7
        x = torch.tensor(x).float()
        return x


class RubiksGraph(Graph):

    def __init__(self, args):
        # TODO: Load model
        self.model = None

    def heuristic(self, states: list, target: State) -> list:
        # This does everything in a single batch on the GPU.
        # We must be careful if this batch gets too large.
        # (i.e., when number of next states can't fit in GPU memory)
        arr = np.asarray([state_to_image(s) for s in states])
        tensor = self.model.predict(arr)
        result = tensor.numpy() * (MAX_SCRAMBLE_LENGTH + 1)
        return [value[0] for value in result]
        #return [0 for _ in states]



