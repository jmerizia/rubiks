from abc import ABC, abstractmethod
from helpers import PriorityQueue
import time
from multiprocessing import Pool
import numpy as np

VERBOSE = True

def get_next_actions_and_states(states):
    """
    Function wrapper around get_next_actions and apply_action,
    so that actions and states can be quickly
    preprocessed in parallel
    """
    res = []
    for state in states:
        next_state_actions = []
        for action in state.get_next_actions():
            next_state_actions.append((action, state.apply_action(action)))
        res.append(next_state_actions)
    return res

def flatten(deep):
    """
    flattens a 2d list
    """
    res = []
    for l in deep:
        res += l
    return res


class Action(ABC):
    """
    This is a data holder for a Graph Action.
    """
    pass


class State(ABC):
    """
    This is a data holder for a Graph State.
    """
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __hash__(self, other):
        """
        Hash this state
        """
        pass

    @abstractmethod
    def __lt__(self, other):
        """
        Compare this state to another.
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """
        Check if this state is equal to another
        """
        pass

    @abstractmethod
    def apply_action(self, a: Action) -> 'State':
        """
        Apply an action to this state, returning a new state.
        """
        pass
        # Note: 'State' is a forward reference, since it appear in the
        #       definition of the State class.

    @abstractmethod
    def get_next_actions(self) -> list:
        """
        Returns the next states to this state.
        """
        pass

    @abstractmethod
    def trainable(self):
        """Returns a torch float tensor for training."""
        pass



class Graph:
    """
    This is a partially implemented class for searching on an
        arbitrary graph with A* search.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def heuristic(self, states: list, target: State) -> list:
        """
        Calculate a heuristic between a list of states and a target state.
        This should be implemented blocking.
        Implement in parallel for good performance.
        """
        pass

    def get_path(self, tree: dict, start: State, target: State) -> list:
        """
        Given a tree, return the path up the tree from
        from target to start, assuming it exists.
        Elements in the resulting list are (state, action) pairs.
        """
        path = []
        cur = target
        while cur != start:
            cur, action = tree[hash(cur)]
            path.append((cur, action))
        return list(reversed(path))


    def connected(self, start: State, target: State) -> list:
        """
        Performs a search over the graph to determine if
            two vertices (start and target) are connected.
        If they are connected,
            this returns a path of (state, action) pairs.
        This is a standard A* algorithm implementation.
        If the given timeout is reached,
            then the function will terminate.
        If timeout is 0, then function will not terminate
            until the result is found, possibly running indefinitely.
        """
        # set of node hashes:
        OPEN, CLOSED = set(), set()
        # priority queue of nodes:
        Q = PriorityQueue()
        # map from node hash -> (node, action)
        parent = dict()
        # map from node hash -> float:
        g, h = dict(), dict() 

        # Initialize
        hash_start = hash(start)
        OPEN.add(hash_start)
        Q.add(value=start, priority=0)
        MAX_BEAM_WIDTH = 200
        WEIGHTING_FACTOR = 0.1
        g[hash_start] = 0
        h[hash_start] = 0
        NUM_THREADS = 8
        P = Pool(NUM_THREADS)
        hash_target = hash(target)
        reopen_count = 0

        cnt = 0
        while not Q.empty():

            # new set of nodes whose heuristic values must be calculated as a batch
            NEW = []

            # Get the top N nodes in OPEN with lowest f
            nodes = Q.popn(MAX_BEAM_WIDTH)
            if VERBOSE:
                print('OPEN:', len(OPEN),
                      'CLOSED:', len(CLOSED),
                      'reopen count:', reopen_count,
                      'total expanded:', len(OPEN) + len(CLOSED))
                print('Expanding {} nodes...'.format(len(nodes)))
            st = time.time()

            # We know we will need all of these next states,
            # so let's preprocess them in parallel
            states = [state for state, priority in nodes]
            if len(states) <= NUM_THREADS:
                PRE = get_next_actions_and_states(states)
            else:
                per_thread = len(states) // NUM_THREADS + 2
                query = [states[i:i+per_thread] for i in range(0, len(states), per_thread)]
                print(len(query), len(query[0]))
                result = P.map(get_next_actions_and_states, query)
                PRE = flatten(result)

            # Process this 'beam' of nodes
            for (u, priority), next_action_states in zip(nodes, PRE):
                hash_u = hash(u)

                # If we reached the target node
                if hash_u == hash_target:
                    return self.get_path(parent, start, target)

                # Consider all next nodes
                for a, v in next_action_states:
                    hash_v = hash(v)

                    # If v is neither closed nor open
                    if hash_v not in CLOSED and hash_v not in OPEN:

                        # Open v and update it's cost and parent
                        OPEN.add(hash_v)
                        NEW.append(v)
                        g[hash_v] = g[hash_u] + 1
                        parent[hash_v] = (u, a)

                    # Else, if the current cost of getting to v
                    # is larger than going from u to v
                    elif g[hash_v] > g[hash_u] + 1:

                        reopen_count += 1

                        # Update the cost and parent of v
                        g[hash_v] = g[hash_u] + 1
                        parent[hash_v] = (u, a)

                        # If v closed, and hence not open
                        if hash_v in CLOSED:

                            # Re-open v, and heuristic must be recalculated
                            CLOSED.remove(hash_v)
                            OPEN.add(hash_v)
                            NEW.append(v)

                # Finished exploring children of u,
                # so now close u
                OPEN.remove(hash_u)
                CLOSED.add(hash_u)

            en = time.time()
            if VERBOSE:
                print('finished expanding', en - st)

            if len(NEW) > 0:

                # Calculate and update heuristics in a batch
                if VERBOSE:
                    print('Calculating heuristic...')
                st = time.time()
                h = self.heuristic(NEW, target)
                for idx, v in enumerate(NEW):
                    hash_v = hash(v)
                    priority = WEIGHTING_FACTOR * g[hash_v] + h[idx]
                    # Note: v might already be in the priority queue,
                    #       since we may have re-opened it.
                    #       Thus, PriorityQueue must support updating weights.
                    Q.add(value=v, priority=priority)
                en = time.time()
                if VERBOSE:
                    print('finished heuristics', en - st)
            else:
                if VERBOSE:
                    print('No new values')

        return None
