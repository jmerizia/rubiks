import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import fire
import wandb
import os
import time
from model import RubiksNetwork
import numpy as np
from rubiks import RubiksState, RubiksAction
from collections import defaultdict

class RubiksDataset:
    def __init__(self, dataset_fname):
        self.states = []
        self.labels = []
        with open(dataset_fname) as f:
            self.l = int(f.readline())
            for i in range(self.l):
                k = int(f.readline())
                state = tuple(map(int, f.readline().split()))
                state = RubiksState(state)
                self.states.append(state)
                self.labels.append(k)
        self.cur_idx = 0

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
                y = 1 + torch.zeros([1], dtype=torch.float).to(device)
            else:
                x = x.trainable().to(device)
                y = 1 + model(x)
            possible.append(y)
        possible = torch.stack(possible, 0)
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


def supervised(model, device, dataset_train, dataset_test, bs, lr, log_step):
    print('[SUP] Running supervised training')
    model.train()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer.zero_grad()
    mseloss = nn.MSELoss()

    m = 1
    while not dataset_train.empty():
        labels, states = dataset_train.get_next(bs)

        # train with back-prop
        st = time.time()
        x = [state.trainable() for state in states]
        x = torch.stack(x, 0)
        x = x.to(device)
        yp = model(x)
        out = mseloss(y, yp)
        out.backward()
        optimizer.step()
        loss = out.item()
        en = time.time()
        train_time = en-st
        
        if m % log_step == 0:
            print('Batch')


def calc_accuracy(model, test_labels, test_states):
    freq = defaultdict(lambda: 0)
    tota = defaultdict(lambda: 0)
    uniq = set()
    target = RubiksState()
    for k, state in zip(test_labels, test_states):
        x = state.trainable().to(device)
        y = model(x).item()
        if abs(y - k) < 0.5:
            freq[k] += 1
        tota[k] += 1
        uniq.add(k)
    for k in uniq:
        acc = freq[k] / tota[k]
        wandb.log({'DAVI-k-{}'.format(k): acc}, step=dataset_train.cur_idx)
        print('[DAVI] acc k-{} {}'.format(k, acc))


def davi(model, device, dataset_train, dataset_test, bs, lr, log_step, check, greedy_test_step, threshold):

    print('[DAVI] Running DAVI')
    test_labels, test_states = dataset_test.get_next(len(dataset_test))
    model.train()
    mseloss = nn.MSELoss()

    # set up copied model
    model_e = RubiksNetwork()
    model_e.load_state_dict(model.state_dict())
    model_e.to(device)
    model_e.eval()

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer.zero_grad()
    m = 1
    while not dataset_train.empty():

        # load batch:
        labels, states = dataset_train.get_next(bs)

        # feed forward (one level deep of tree):
        st = time.time()
        y = [predict(model_e, device, state) for state in states]
        y = torch.stack(y, 0).numpy()
        en = time.time()
        pred_time = en-st

        # train with back prop:
        st = time.time()
        x = [state.trainable() for state in states]
        x = torch.stack(x, 0)
        x = x.to(device)
        yp = model(x)
        out = mseloss(y, yp)
        out.backward()
        optimizer.step()
        loss = out.item()
        en = time.time()
        train_time = en-st

        # update weights:
        if m % check == 0:
            if loss < threshold:
                print('[DAVI] updating!')
                model_e.load_state_dict(model.state_dict())
                wandb.log({'DAVI-update': 1}, step=dataset_train.cur_idx)
            else:
                wandb.log({'DAVI-update': 0}, step=dataset_train.cur_idx)

        # perform test
        if m % log_step == 0:
            print('[DAVI] batch {}, data: {}, loss = {}'.format(m, dataset_train.cur_idx, loss))
            print('[DAVI] pred_time = {:0.4f}, train_time = {:0.4f}' \
                    .format(setup_time, pred_time, train_time))
            wandb.log({'DAVI-loss': loss}, step=dataset_train.cur_idx)


            print('[DAVI] Performing test...')
            uniq = calc_accuracy(model, test_labels, test_states)


        if m % greedy_test_step == 0:
            correct = 0
            for state in test_states:
                if greedy_search(model, device, state, target, 10):
                    correct += 1
            print('[DAVI] Greedy Search Result: {}/{} ({})'.format(correct, len(test_states), en-st))
            acc = correct / len(test_states)
            wandb.log({'DAVI-greedy-search': acc}, step=dataset_train.cur_idx)

        m += 1


def entry(sup_lr=0.005,
          davi_lr=0.005,
          sup_bs=400,
          davi_bs=400,
          sup_log_step=20,
          davi_log_step=20,
          check=100,
          threshold=0.05,
          greedy_test_step=50,
          seed=1,
          cuda=True):

    wandb.init(project='rubiks')

    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")

    print('Loading datasets:')
    dataset_train_davi = RubiksDataset('./data/davi-train.txt')
    dataset_train_sup  = RubiksDataset('./data/sup-train.txt')
    dataset_test_davi  = RubiksDataset('./data/test.txt')
    dataset_test_sup   = RubiksDataset('./data/test.txt')
    en = time.time()
    print('Loaded dataset in {:0.4f} seconds'.format(en-st))

    supervised(model=model,
               device=device,
               dataset_train=None,
               dataset_test=None,
               bs=sup_bs,
               lr=sup_lr,
               log_step=sup_log_step)
    
    davi(model=model,
         device=device,
         dataset_train=None,
         dataset_test=None,
         bs=davi_bs,
         lr=davi_lr,
         log_step=davi_log_step,
         check=check,
         greedy_test_step=greedy_test_step,
         threshold=threshold)

    torch.save(state_dict, 'model-weights.pt')


if __name__ == '__main__':
    fire.Fire(entry)

