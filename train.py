import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import fire
import wandb
import os
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
    possible = []
    for a in state.get_next_actions():
        x = state.apply_action(a).to_numpy()
        x = torch.tensor(x).to(device)
        yh = 1 + model(x)
        possible.append(yh)
    possible = torch.stack(possible, 0)
    ans = torch.min(possible, 0)
    return ans


def davi(device, batch_size, learning_rate, convergence_check, error_threshold):
    dataset = RubiksDataset('./data/dataset.txt')
    mseloss = nn.MSELoss()
    theta = None
    theta_e = None
    m = 1
    while True:

        # initialize model parameters:
        model = RubiksNetwork()
        if theta:
            model.load_state_dict(theta)
        model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

        # load batch:
        states = dataset.get_next(batch_size)

        # feed forward (one level deep of tree):
        y = []
        for state in states:
            predict(model, device, state)
        y = torch.cat(y)

        # set up training:
        model.train()
        optimizer.zero_grad()

        # train with back prop:
        x = [state.to_numpy() for state in states]
        x = [torch.tensor(state) for state in states]
        x = torch.stack(x, 0)
        x = x.to(device)
        yp = model(x)
        out = mseloss(y, yp)
        out.backward()
        optimizer.step()
        loss = out.item()

        # update weights:
        theta = model.state_dict()
        if m % convergence_check and loss < error_threshold:
            theta_e = model.state_dict()

        m += 1


def entry(use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    davi(device=device,
         batch_size=1,
         learning_rate=0.01,
         convergence_check=10,
         error_threshold=0.01)


if __name__ == '__main__':
    fire.Fire(entry)

