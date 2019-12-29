from heapq import heappush, heappop, heapify
from abc import ABC, abstractmethod
from multiprocessing import Pool


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


class Graph:
    """
    This is a partially implemented class for searching on an
        arbitrary graph with A* search.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_next_actions(self, s: State) -> list:
        """
        Returns the next states in the graph.
        """
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


    def connected(self, start: State, target: State, timeout=0) -> list:
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
        # TODO: Use timeout
        vis = set()
        PQ = []
        parent = dict()
        vis.add(hash(start))
        heappush(PQ, (0, start))
        p = Pool(2)
        while PQ:
            priority, cur_state = heappop(PQ)
            next_actions = self.get_next_actions(cur_state)
            # Pre-compute all of the heuristic values.
            # Note: heuristic() should be implemented in parallel for good performance.
            heuristics = self.heuristic(
                    [cur_state.apply_action(a) for a in next_actions],
                    target)

            for action, heuristic in zip(next_actions, heuristics):
                # TODO: We may want to prune these next actions
                #       to a fixed quantity.
                #       We probably shouldn't consider every
                #       next step, just the top most probable ones.
                state = cur_state.apply_action(action)
                state_hash = hash(state)
                if state == target:
                    parent[state_hash] = (cur_state, action)
                    print('Found path after searching {} states'.format(len(vis) + 1))
                    return self.get_path(parent, start, target)
                if state_hash not in vis:
                    parent[state_hash] = (cur_state, action)
                    vis.add(state_hash)
                    new_priority = priority + 1 + heuristic
                    heappush(PQ, (new_priority, state))
                if len(vis) % 200 == 0:
                    print('Checked {} states'.format(len(vis)))

        # In practice, this line should never be run,
        # since the search space will be very large.
        # I leave it here for correctness.
        return None
