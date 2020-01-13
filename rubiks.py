from abstract import State, Action, Graph
import random
import os
import pickle
import time
import numpy as np
from multiprocessing import Pool
from models.CNN import CNN
import argparse

EXAMPLE_CACHE_FNAME = os.path.join(
        os.path.dirname(__file__), 
        'cache_files/rubiks_cache.in')

MODEL_CHECKPOINT_DIR = os.path.join(
        os.path.dirname(__file__), 
        'model_checkpoints/')

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
        return str(self.perm)

    def __repr__(self):
        return str(self.perm)

    def apply_action(self, a: RubiksAction) -> 'RubiksState':
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


def random_scramble(k: int) -> RubiksState:
    """
    Returns a random scramble of distance no greater
    than k from the solved state.
    """
    state = RubiksState()
    for _ in range(k):
        action = RubiksAction(random.choice(RUBIKS_ACTIONS))
        state = state.apply_action(action)
    return state

def generate_examples_helper(k: int) -> list:
    # It makes sense that the number of scrambles must
    # increase, as the number of steps away increases.
    # This number can be tweaked to give more,
    # or fewer scrambles.
    num_scrambles = 50*k*k + 50
    return [(k, random_scramble(k)) for _ in range(num_scrambles)]

def generate_examples(k: int) -> list:
    """
    Returns a list of example scrambles
    for each of the 0 to k possible length
    scrambles.
    Each scrambled state is listed in a pair
    with the associated k value.
    """
    p = Pool(20)
    examples = []
    for example in p.map(generate_examples_helper, range(k+1)):
        examples += example
    return examples

def state_to_image(state: RubiksState):
    # Just a neat trick:
    t = [0] + [RUBIKS_COLORS[k] / 7.0 for k in state.perm]
    # TODO: Experiment with different arrangements of these "pixels"
    return np.array(
        [t[ 1], t[ 2], t[ 3], t[ 4], t[ 5], t[ 6], t[ 7], t[ 8], t[ 9], t[ 1], t[ 2], t[ 3],
         t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18], t[10], t[11], t[12],
         t[19], t[20], t[21], t[22], t[23], t[24], t[25], t[26], t[27], t[19], t[20], t[21],
         t[28], t[29], t[30], t[31], t[32], t[33], t[34], t[35], t[36], t[28], t[29], t[30],
         t[37], t[38], t[39], t[40], t[41], t[42], t[43], t[44], t[45], t[37], t[38], t[39],
         t[46], t[47], t[48], t[49], t[50], t[51], t[52], t[53], t[54], t[46], t[47], t[48],
        # Just double the image:
         t[ 1], t[ 2], t[ 3], t[ 4], t[ 5], t[ 6], t[ 7], t[ 8], t[ 9], t[ 1], t[ 2], t[ 3],
         t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18], t[10], t[11], t[12],
         t[19], t[20], t[21], t[22], t[23], t[24], t[25], t[26], t[27], t[19], t[20], t[21],
         t[28], t[29], t[30], t[31], t[32], t[33], t[34], t[35], t[36], t[28], t[29], t[30],
         t[37], t[38], t[39], t[40], t[41], t[42], t[43], t[44], t[45], t[37], t[38], t[39],
         t[46], t[47], t[48], t[49], t[50], t[51], t[52], t[53], t[54], t[46], t[47], t[48]]
    )

def generate_or_load_dataset() -> tuple:
    """
    Generate/load dataset, depending on it a cache_files/rubiks_cache.in
    file can be found.
    Returns a tuple of four numpy array of shape (N, width, height, 1).
    The tuple contains the following:
        0. train images
        1. train labels
        2. test images
        3. test labels
    """

    MAX_LENGTH = 19

    if os.path.exists(EXAMPLE_CACHE_FNAME):
        print('Cache found, loading examples from cache')
        with open(EXAMPLE_CACHE_FNAME, 'rb') as f:
            examples = pickle.load(f)
        print('Loaded {} examples from cache'.format(len(examples)))
    else:
        print('No cache found, generating examples. May take a while...')
        examples = generate_examples(MAX_LENGTH)
        with open(EXAMPLE_CACHE_FNAME, 'wb') as f:
            pickle.dump(examples, f)
        print('Generated {} examples and saved in cache'.format(len(examples)))

    print('Splitting and shaping the examples')
    # Split the data
    num_train = int(len(examples)*0.80)
    num_test  = int(len(examples)*0.20)
    random.shuffle(examples)
    examples_train = examples[:num_train]
    examples_test  = examples[:num_test]
    
    # Shape the data
    labels_train = np.asarray([float(d) / (MAX_LENGTH + 1) for d, s in examples_train])
    labels_test  = np.asarray([float(d) / (MAX_LENGTH + 1) for d, s in examples_test ])
    images_train = np.asarray([state_to_image(s) for d, s in examples_train])
    images_test  = np.asarray([state_to_image(s) for d, s in examples_test])

    labels_train = labels_train.reshape((-1, 1))
    labels_test  = labels_test.reshape((-1, 1))

    return (images_train, labels_train, images_test, labels_test)


def train_or_load_model(args):
    model_name    = args.model_name
    batch_size    = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    dropout_rate  = float(args.dropout_rate)
    epochs        = int(args.epochs)
    model_fname = os.path.join(MODEL_CHECKPOINT_DIR, model_name)
    if os.path.exists(model_fname + '.index'):
        print('Found saved model -- loading it')
        # load model
        model = CNN(model_fname=model_fname,
                    model_name=model_name,
                    batch_size=batch_size,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    epochs=epochs)
    else:
        print('No model "{}" found. Training one instead'.format(model_fname))
        # generate data and train model
        images_train, labels_train, images_test, labels_test = generate_or_load_dataset()
        #print('images_train', images_train.shape)
        #print('labels_train', labels_train.shape)
        #print('images_test', images_test.shape)
        #print('labels_test', labels_test.shape)
        model = CNN(model_name=model_name,
                    batch_size=batch_size,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate,
                    epochs=epochs)
        st = time.time()
        model.train(
            train_x=images_train,
            train_y=labels_train,
            test_x=images_test,
            test_y=labels_test
        )
        en = time.time()
        print("Trained model! elapsed time:", en - st)
        model.save(model_fname)

    return model


class RubiksGraph(Graph):

    def __init__(self, args):
        self.model = train_or_load_model(args)

    def get_next_actions(self, s: RubiksState) -> list:
        actions = [RubiksAction(a) for a in RUBIKS_ACTIONS]
        random.shuffle(actions)
        return actions

    def heuristic(self, states: list, target: State) -> list:
        # This does everything in a single batch on the GPU.
        # We must be careful if this batch gets too large.
        # (i.e., when number of next states can't fit in GPU memory)
        tensor = self.model.predict(list(map(state_to_image, states))) * 7
        return [result.numpy()[0] for result in tensor]
        #return [0 for _ in states]


parser = argparse.ArgumentParser(description='Train and solve Rubiks Cubes.')
parser.add_argument('--model-name', dest='model_name',
        required=True, help='Name of the model to train')
parser.add_argument('--epochs', dest='epochs',
        required=True, help='Number of epochs to train for')
parser.add_argument('--learning-rate', dest='learning_rate',
        required=True, help='Learning rate for training')
parser.add_argument('--batch-size', dest='batch_size',
        required=True, help='Batch size for training')
parser.add_argument('--dropout-rate', dest='dropout_rate',
        required=True, help='Dropout rate for training')
args = parser.parse_args()

graph  = RubiksGraph(args)
quit()
# Target is the solved state
target = RubiksState()
# Start at some scrambled state:
scramble = 'F* R* U* B* D* B* R* D*'.split()
scramble = 'F* R* U* B*'.split()
scramble = 'F* R* U* B* D* B2 D2'.split()
start = RubiksState()
for action in scramble:
    start = start.apply_action(RubiksAction(action))

path = graph.connected(start, target)
if path is None:
    print('No path')
elif path == []:
    if start != target:
        print('path == [], but expected start == target')
    print('Already solved (start == target)')
else:
    print('Solution:', ' '.join(a.name for s, a in path))
