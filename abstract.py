from heapq import heappush, heappop, heapify
from abc import ABC, abstractmethod

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
        Returns the next states of 
        """
        pass

    @abstractmethod
    def apply_action(self, a: Action, s: State) -> State:
        """
        Apply an action to this state, returning a new state.
        """
        pass

    @abstractmethod
    def heuristic(self, s1: State, s2: State) -> float:
        """
        A heuristic between two states.
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
        while PQ:
            priority, cur_state = heappop(PQ)
            for action in self.get_next_actions(cur_state):
                # TODO: We may want to prune these next actions
                #       to a fixed quantity.
                #       We probably shouldn't consider every
                #       next step, just the top most probable ones.
                state = self.apply_action(action, cur_state)
                state_hash = hash(state)
                parent[state_hash] = (cur_state, action)
                if state == target:
                    return self.get_path(parent, start, target)
                if state_hash not in vis:
                    vis.add(state_hash)
                    new_priority = priority + 1 + self.heuristic(state, target)
                    heappush(PQ, (new_priority, state))

        # In practice, this line should never be run,
        # since the search space will be very large.
        # I leave it here for correctness.
        return None
