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


class RubiksDataset:
    def __init__(self, dataset_fname):
        self.states = []
        with open(dataset_fname) as f:
            self.l = int(f.readline())
            for line in f:
                state = tuple(map(int, line.split()))
                state = RubiksState(state)
                self.states.append(state)
        self.cur_idx = 0

    def __len__(self):
        return self.l

    def get_next(self, n):
        res = []
        for i in range(n):
            k = i + self.cur_idx
            if k >= len(self.states):
                break
            res.append(self.states[k])
        self.cur_idx += len(res)
        return res

    def empty(self):
        return self.l == self.cur_idx


def predict(model, device, state):
    """Given a state and model, predict the heuristic."""

    if state == RubiksState():
        ans = torch.zeros([1], dtype=torch.float)

    else:
        possible = []
        for a in state.get_next_actions():
            x = state.apply_action(a)
            if x == RubiksState():
                y = torch.zeros([1], dtype=torch.float)
            else:
                x = x.trainable().to(device)
                y = 1 + model(x)
            possible.append(y)
        possible = torch.stack(possible, 0)
        ans = torch.min(possible, 0)[0]

    return ans


def greedy_search(model, device, start, target, depth):
    state = start
    for i in range(depth):

        # we reached the target:
        if state == target:
            return i

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

    return -1


def davi(device, bs, lr, check, test_step, threshold, state_dict=None):
    dataset = RubiksDataset('./data/dataset.txt')
    mseloss = nn.MSELoss()
    theta = state_dict
    theta_e = state_dict
    m = 1
    while not dataset.empty():

        # initialize model parameters:
        st = time.time()
        model = RubiksNetwork()
        model_e = RubiksNetwork()
        if theta:
            model.load_state_dict(theta)
        if theta_e:
            model_e.load_state_dict(theta_e)
        model.to(device)
        model_e.to(device)
        # we do not optimize model_e (it's just used for ff)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)

        # load batch:
        states = dataset.get_next(bs)

        en = time.time()
        setup_time = en-st

        # feed forward (one level deep of tree):
        st = time.time()
        model_e.eval()
        y = []
        for state in states:
            yi = predict(model_e, device, state)
            y.append(yi)
        y = torch.stack(y, 0)
        en = time.time()
        pred_time = en-st

        # set up training:
        model.train()
        optimizer.zero_grad()

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
        theta = model.state_dict()
        if m % check == 0:
            print('batch {}: loss = {}'.format(m, loss))
            print('setup_time = {:0.4f}, pred_time = {:0.4f}, train_time = {:0.4f}' \
                    .format(setup_time, pred_time, train_time))
            if loss < threshold:
                print('updating!')
                theta_e = model.state_dict()

                print('Performing test...')
                test_states = [
                    (0, RubiksState()),
                    (1, RubiksState() \
                        .apply_action(RubiksAction('R')) ),
                    (1, RubiksState() \
                        .apply_action(RubiksAction('L')) ),
                    (1, RubiksState() \
                        .apply_action(RubiksAction('L2')) ),
                    (2, RubiksState() \
                        .apply_action(RubiksAction('R')) \
                        .apply_action(RubiksAction('D')) ),
                    (2, RubiksState() \
                        .apply_action(RubiksAction('B')) \
                        .apply_action(RubiksAction('F2')) ),
                    (3, RubiksState() \
                        .apply_action(RubiksAction('B')) \
                        .apply_action(RubiksAction('F2')) \
                        .apply_action(RubiksAction('D')) ),
                    (3, RubiksState() \
                        .apply_action(RubiksAction('B')) \
                        .apply_action(RubiksAction('D')) \
                        .apply_action(RubiksAction('R')) ),
                    (3, RubiksState() \
                        .apply_action(RubiksAction('D')) \
                        .apply_action(RubiksAction('R2')) \
                        .apply_action(RubiksAction('F*')) )
                    #RubiksState().apply_action(RubiksAction('D')),
                    #RubiksState().apply_action(RubiksAction('U')),
                    #RubiksState().apply_action(RubiksAction('F')),
                    #RubiksState().apply_action(RubiksAction('B')),
                    #RubiksState().apply_action(RubiksAction('L')),
                    #RubiksState().apply_action(RubiksAction('R')),
                    #RubiksState().apply_action(RubiksAction('D*')),
                    #RubiksState().apply_action(RubiksAction('U*')),
                    #RubiksState().apply_action(RubiksAction('F*')),
                    #RubiksState().apply_action(RubiksAction('B*')),
                    #RubiksState().apply_action(RubiksAction('L*')),
                    #RubiksState().apply_action(RubiksAction('R*')),
                    #RubiksState().apply_action(RubiksAction('D2')),
                    #RubiksState().apply_action(RubiksAction('U2')),
                    #RubiksState().apply_action(RubiksAction('F2')),
                    #RubiksState().apply_action(RubiksAction('B2')),
                    #RubiksState().apply_action(RubiksAction('L2')),
                    #RubiksState().apply_action(RubiksAction('R2'))
                ]
                target = RubiksState()
                correct = 0
                st = time.time()
                for ans, state in test_states:
                    x = state.trainable().to(device)
                    y = model(x).item()
                    print(ans, y)
                    #res = greedy_search(model, device, state, target, 30)
                    #if res > -1:
                    #    correct += 1
                en = time.time()
                #print('Result: {}/{} ({})'.format(correct, len(test_states), en-st))

        m += 1


def entry(lr,
          bs,
          check,
          threshold,
          test_step,
          seed=1,
          epochs=1,
          cuda=False):

    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")

    state_dict = None
    for i in range(epochs):
        state_dict = davi(device=device,
                          bs=bs,
                          lr=lr,
                          check=check,
                          test_step=test_step,
                          threshold=threshold,
                          state_dict=state_dict)
    torch.save(state_dict, 'model-weights.pt')


if __name__ == '__main__':
    fire.Fire(entry)

