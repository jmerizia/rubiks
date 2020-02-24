from rubiks import RubiksAction, RubiksState, RubiksGraph
import cProfile
import fire
import time
import os

def entry(profile=False):
    if profile:
        cProfile.run('graph.connected(start, target)', 'stats')

    else:
        args = {}
        graph = RubiksGraph(args)

        st = time.time()
        path = graph.connected(start, target)
        en = time.time()
        print('time:', en - st)

        if path is None:
            print('No path')
        elif path == []:
            if start != target:
                print('path == [], but expected start == target')
            print('Already solved (start == target)')
        else:
            print('Solution:', ' '.join(a.name for s, a in path))
            print('hey')


if __name__ == '__main__':
    fire.Fire(entry)
