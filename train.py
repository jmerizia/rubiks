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
from multiprocessing import Pool
INF = float('inf')


class RubiksDataset:
    def __init__(self, dataset_fname, max_data=INF):
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


def supervised(model, device, dataset, epoch, bs, lr, log_step):
    print('[SUP] Running supervised training')
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer.zero_grad()
    mseloss = nn.MSELoss()

    m = 1
    while not dataset.empty():

        # collect batch
        labels, states = dataset.get_next(bs)

        # train with back-prop
        st = time.time()
        y = torch.tensor(labels)
        y = y.to(device)
        y = y.unsqueeze(1)
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
            print('[SUP] epoch: {}, batch: {}, data: {}, loss: {:0.4f}'.format(epoch, m, dataset.cur_idx, loss))
            print('[SUP] train_time = {:0.4f}'.format(train_time))

        m += 1

    return loss


def davi(model, device, dataset_train, dataset_test, step, bs, lr, log_step, check, greedy_test_step, threshold):

    print('[DAVI] Running DAVI')
    test_labels, test_states = dataset_test.get_next(len(dataset_test))
    mseloss = nn.MSELoss()

    # set up copied model
    model_e = RubiksNetwork()
    model_e.load_state_dict(model.state_dict())
    model_e.to(device)
    model_e.eval()

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    m = 1
    while not dataset_train.empty():

        # load batch:
        labels, states = dataset_train.get_next(bs)
        step += len(states)

        # feed forward (one level deep of tree):
        st = time.time()
        y = [predict(model_e, device, state) for state in states]
        y = torch.stack(y, 0)
        en = time.time()
        pred_time = en-st

        # train with back prop:
        model.train()
        st = time.time()
        optimizer.zero_grad()
        # computing x and moving to device is actually quite fast...
        x = [state.trainable() for state in states]  
        x = torch.stack(x, 0)
        x = x.to(device)
        yp = model(x)
        out = mseloss(y, yp)
        # the majority of the time is spent doing backprop...
        out.backward()
        optimizer.step()
        loss = out.item()
        en = time.time()
        train_time = en-st

        # update weights:
        if m % check == 0:
            print('[DAVI] Checking for convergence...')
            if loss < threshold:
                print('[DAVI] updating!')
                model_e.load_state_dict(model.state_dict())
                model_e.eval()
                wandb.log({'DAVI-update': 1}, step=step)
            else:
                wandb.log({'DAVI-update': 0}, step=step)

        # perform test
        if m % log_step == 0:
            model.eval()
            print('[DAVI] batch {}, step: {}, loss = {}'.format(m, step, loss))
            wandb.log({'DAVI-loss': loss}, step=step)

            print('[DAVI] Performing test...')
            accs = calc_accuracy(model, device, test_labels, test_states)
            for k, acc in accs.items():
                wandb.log({'DAVI-k-{}'.format(k): acc}, step=step)
                print('[DAVI] acc k-{} {}'.format(k, acc))

        # Check greedy solve
        #if m % greedy_test_step == 0:
        #    model.eval()
        #    correct = 0
        #    target = RubiksState()
        #    for state in test_states:
        #        if greedy_search(model, device, state, target, 10):
        #            correct += 1
        #    print('[DAVI] Greedy Search Result: {}/{} ({})'.format(correct, len(test_states), en-st))
        #    acc = correct / len(test_states)
        #    wandb.log({'DAVI-greedy-search': acc}, step=dataset_train.cur_idx)
        #    model.train()

        print('[DAVI] Finished batch {}, loss = {:0.4f}'.format(m, loss))
        print('[DAVI] pred_time = {:0.4f}, train_time = {:0.4f}'.format(pred_time, train_time))
        m += 1

    return step


def entry(
          # supervised parameters
          sup_epochs=50,
          sup_lr=0.008,
          sup_bs=5000,
          sup_log_step=10, # how often to print to console
          # davi parameters
          davi_epochs=1,
          davi_lr=0.01,
          davi_bs=200,
          davi_log_step=10, # how often to print to console and wandb (# batches)
          check=20, # how often to check for convergence (# batches)
          threshold=0.05, # threshold for convergence
          greedy_test_step=20, # how often to perform greedy search test (# batches)
          skip_sup=False,
          seed=1,
          cuda=True):

    wandb.init(project='rubiks')

    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")

    model = RubiksNetwork()
    model.to(device)

    if not skip_sup:

        print('[SUP] Loading datasets:')
        st = time.time()
        dataset_train_sup  = RubiksDataset('./data/sup-train.txt')
        dataset_test_sup   = RubiksDataset('./data/test.txt')
        en = time.time()
        print('[SUP] Loaded datasets in {:0.4f} seconds'.format(en-st))

        # train in supervised fashion
        for epoch in range(1, sup_epochs+1):

            model.train()
            loss = supervised(model=model,
                              device=device,
                              dataset=dataset_train_sup,
                              epoch=epoch,
                              bs=sup_bs,
                              lr=sup_lr,
                              log_step=sup_log_step)

            wandb.log({'SUP-loss': loss}, step=epoch)
            model.eval()
            test_labels, test_states = dataset_test_sup.get_next(len(dataset_test_sup))
            accs = calc_accuracy(model, device, test_labels, test_states)
            msg = ''
            for k, acc in accs.items():
                msg += '{}: {:0.4f}, '.format(k, acc)
                wandb.log({'SUP-k-{}'.format(k): acc}, step=epoch)
            print('[SUP] epoch {} ->'.format(epoch), msg)
            dataset_train_sup.reset()
            dataset_test_sup.reset()

        # save checkpoint
        torch.save(model.state_dict(), 'model-sup.pt')

    else:

        pass
        #print('Loading supervised-trained model instead...')
        #model.load_state_dict(torch.load('model-sup.pt'))

    davi_training_datasets = [
        ('./data/davi-1.txt', 3),
        ('./data/davi-2.txt', 1),
        ('./data/davi-3.txt', 1),
        ('./data/davi-4.txt', 1),
        ('./data/davi-5.txt', 1),
        ('./data/davi-6.txt', 1)
    ]

    step = 0
    for fname, epochs in davi_training_datasets:

        for i in range(1, epochs+1):

            print('[DAVI] Loading dataset', fname)
            st = time.time()
            dataset_train_davi = RubiksDataset(fname)
            dataset_test_davi  = RubiksDataset('./data/test.txt')
            en = time.time()
            print('[DAVI] Loaded dataset in {:0.4f} seconds'.format(en-st))

            # train in reinforcement fashion
            step = davi(model=model,
                        device=device,
                        dataset_train=dataset_train_davi,
                        dataset_test=dataset_test_davi,
                        step=step,
                        bs=davi_bs,
                        lr=davi_lr,
                        log_step=davi_log_step,
                        check=check,
                        greedy_test_step=greedy_test_step,
                        threshold=threshold)

        torch.save(model.state_dict(), 'model-davi.pt')


if __name__ == '__main__':
    fire.Fire(entry)

