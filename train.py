import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import fire
import os
import time
import wandb
from model import RubiksNetwork
import numpy as np
import random
from rubiks import RubiksState, RubiksAction
from collections import defaultdict
from multiprocessing import Pool
INF = float('inf')
global_step = 1

class RubiksDataset:
    def __init__(self, dataset_fname, max_data=INF, shuffle=False):
        self.states = []
        self.labels = []
        self.shuffle = shuffle
        with open(dataset_fname) as f:
            self.l = min(max_data, int(f.readline()))
            for i in range(self.l):
                k = int(f.readline())
                state = tuple(map(int, f.readline().split()))
                state = RubiksState(state)
                self.states.append(state)
                self.labels.append(k)
        self.cur_idx = 0
        if self.shuffle:
            comb = list(zip(self.labels, self.states))
            random.shuffle(comb)
            self.labels, self.states = zip(*comb)
        self.labels = list(self.labels)
        self.states = list(self.states)

    def __len__(self):
        return self.l

    def get_next(self, n):
        states = []
        labels = []
        for i in range(n):
            j = i + self.cur_idx
            if j >= self.l:
                break
            states.append(self.states[j])
            labels.append(self.labels[j])
        self.cur_idx += len(states)
        return labels, states

    def empty(self):
        return self.l == self.cur_idx

    def reset(self):
        self.cur_idx = 0
        if self.shuffle:
            comb = list(zip(self.labels, self.states))
            random.shuffle(comb)
            self.labels, self.states = zip(*comb)


def bfs1d(model, device, state):
    """Return the minimum policy of all possible next states plus 1."""

    # I think this makes convergence faster:
    #if state == RubiksState():
    #    return torch.zeros([1], dtype=torch.float).to(device)

    #for a in state.get_next_actions():
    #    x = state.apply_action(a)
    #    if x == RubiksState():
    #        return 1 + torch.zeros([1], dtype=torch.float).to(device)

    model.eval()
    possible = []
    for a in state.get_next_actions():
        x = state.apply_action(a)
        x = x.trainable().to(device)
        x = x.unsqueeze(0)
        y = 1 + model(x)
        possible.append(y)
    possible = torch.cat(possible, 0)
    ans, ind = torch.min(possible, 0)
    return ans


def greedy_search(model, device, start, target, depth):
    state = start
    for i in range(depth):

        # we reached the target:
        if state == target:
            return True

        # figure out which action is the best next one greedily:
        next_state = None
        cur_y = float('inf')
        for action in state.get_next_actions():
            cur_state = state.apply_action(action)
            x = cur_state.trainable().to(device)
            y = model(x).item()
            if y < cur_y:
                cur_y = y
                next_state = cur_state

        # set the best next state:
        state = next_state

    return False


def calc_accuracy(model, device, test_labels, test_states):
    model.eval()
    freq = defaultdict(lambda: 0)
    tota = defaultdict(lambda: 0)
    uniq = set()
    acc = dict()
    for k, state in zip(test_labels, test_states):
        x = state.trainable().to(device)
        x = x.unsqueeze(0)
        y = model(x).item()
        if abs(y - k) < 0.5:
            freq[k] += 1
        tota[k] += 1
        uniq.add(k)
    for k in uniq:
        acc[k] = freq[k] / tota[k]
    return acc


def davi(model, device, optim, dataset, eval_labels, eval_states, bs):
    global global_step

    m = 1
    total_loss = 0
    while not dataset.empty():

        # load batch:
        labels, states = dataset.get_next(bs)

        # feed forward (one level deep of tree):
        st = time.time()
        model.eval()
        y = [bfs1d(model, device, state) for state in states]
        y = torch.stack(y, 0)
        en = time.time()
        pred_time = en-st

        # train with back prop:
        st = time.time()
        model.train()
        optim.zero_grad()
        # computing x and moving to device is actually quite fast...
        x = [state.trainable() for state in states]  
        x = torch.stack(x, 0)
        x = x.to(device)
        yp = model(x)
        out = F.mse_loss(y, yp)
        # the majority of the time is spent doing backprop...
        # (I'm sure my GPU cores are getting saturated)
        out.backward()
        optim.step()
        loss = out.item()
        en = time.time()
        train_time = en-st

        total_loss += loss
        print('[DAVI] Finished batch {}, LOSS = {:0.4f}'.format(m, loss))
        print('[DAVI] pred_time = {:0.2f}, train_time = {:0.2f}'.format(pred_time, train_time))
        m += 1
        global_step += 1

    return total_loss


def entry(epochs=200,
          lr=0.05,
          bs=200,
          seed=1,
          cuda=True,
          log=True,
          use_last_model=False):
    global global_step

    wandb.init(project='rubiks')

    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")

    model = RubiksNetwork()
    if use_last_model:
        model.load_state_dict(torch.load('model-davi.pt'))
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    print('[DAVI] Loading datasets')
    st = time.time()
    dataset_train = RubiksDataset('./data/davi.txt', shuffle=True)
    dataset_test  = RubiksDataset('./data/test.txt', shuffle=True)
    dataset_eval  = RubiksDataset('./data/eval.txt', shuffle=True)
    test_labels, test_states = dataset_test.get_next(len(dataset_test))
    eval_labels, eval_states = dataset_eval.get_next(len(dataset_eval))
    en = time.time()
    print('[DAVI] Loaded datasets in {:0.4f} seconds'.format(en-st))
    print('[DAVI]  * Train = {} '.format(len(dataset_train)))
    print('[DAVI]  * Test  = {} '.format(len(dataset_test)))
    print('[DAVI]  * Eval  = {} '.format(len(eval_states)))

    for epoch in range(1, epochs+1):

        print('[DAVI] Training')
        loss = davi(model=model,
                    device=device,
                    optim=optimizer,
                    dataset=dataset_train,
                    eval_labels=eval_labels,
                    eval_states=eval_states,
                    bs=bs)

        print('[DAVI] Overall loss = {:0.4f}'.format(loss))
        wandb.log({'LOSS': loss}, step=global_step)

        print('[DAVI] Testing')
        accs = calc_accuracy(model, device, test_labels, test_states)
        for k, acc in accs.items():
            wandb.log({'ACC-k-{}'.format(k): acc}, step=global_step)
            print('[DAVI] acc k-{} {}'.format(k, acc))

        model.eval()
        for i, (label, state) in enumerate(zip(eval_labels, eval_states)):
            h = hex(hash(state) % (1<<32))
            x = state.trainable().to(device)
            x = x.unsqueeze(0)
            y = float(model(x).item())
            wandb.log({'EVAL-STATE-{}-{}'.format(h, label): y}, step=global_step)

        torch.save(model.state_dict(), 'model-davi.pt')

        dataset_train.reset()
        dataset_test.reset()


if __name__ == '__main__':
    fire.Fire(entry)

