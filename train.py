import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import fire
import os
import time
from model import RubiksNetwork
import numpy as np
import random
from rubiks import RubiksState, RubiksAction
from collections import defaultdict
from multiprocessing import Pool
INF = float('inf')

class RubiksDataset:
    def __init__(self, dataset_fname, max_data=INF, shuffle=False):
        self.states = []
        self.labels = []
        with open(dataset_fname) as f:
            self.l = min(max_data, int(f.readline()))
            for i in range(self.l):
                k = int(f.readline())
                state = tuple(map(int, f.readline().split()))
                state = RubiksState(state)
                self.states.append(state)
                self.labels.append(k)
        self.cur_idx = 0
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


def predict(model, device, state):
    """Given a state and model, predict the heuristic."""

    if state == RubiksState():
        ans = torch.zeros([1], dtype=torch.float).to(device)

    else:
        possible = []
        for a in state.get_next_actions():
            x = state.apply_action(a)
            if x == RubiksState():
                y = 1 + torch.zeros([1, 1], dtype=torch.float).to(device)
            else:
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


def davi(model, device, optim, dataset, bs):

    m = 1
    while not dataset.empty():

        # load batch:
        labels, states = dataset.get_next(bs)

        # feed forward (one level deep of tree):
        st = time.time()
        model.eval()
        y = [predict(model, device, state) for state in states]
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
        out.backward()
        optim.step()
        loss = out.item()
        en = time.time()
        train_time = en-st

        print('[DAVI] Finished batch {}, LOSS = {:0.4f}'.format(m, loss))
        print('[DAVI] pred_time = {:0.2f}, train_time = {:0.2f}'.format(
            pred_time, train_time))
        m += 1


def entry(epochs=10,
          lr=0.01,
          bs=200,
          seed=1,
          cuda=True,
          log=True):

    wandb.init(project='rubiks')

    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")

    model = RubiksNetwork()
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):

        print('[DAVI] Loading datasets')
        st = time.time()
        dataset_train = RubiksDataset('./data/davi.txt', shuffle=True)
        dataset_test  = RubiksDataset('./data/test.txt', shuffle=True)
        test_labels, test_states = dataset_test.get_next(len(dataset_test))
        en = time.time()
        print('[DAVI] Loaded datasets in {:0.4f} seconds'.format(en-st))

        print('[DAVI] Training')
        davi(model=model,
             device=device,
             optim=optimizer,
             dataset=dataset_train,
             bs=bs)

        print('[DAVI] Testing')
        accs = calc_accuracy(model, device, test_labels, test_states)
        for k, acc in accs.items():
            wandb.log({'DAVI-k-{}'.format(k): acc}, step=epoch)
            print('[DAVI] acc k-{} {}'.format(k, acc))

        torch.save(model.state_dict(), 'model-davi.pt')


if __name__ == '__main__':
    fire.Fire(entry)

