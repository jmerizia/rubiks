from abc import ABC, abstractmethod
from helpers import PriorityQueue
import time


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
            cur, action = tree[cur]
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
        # set of nodes:
        OPEN, CLOSED = set(), set()
        # priority queue of nodes:
        Q = PriorityQueue()
        # map from node to (node, action)
        parent = dict()
        # map from node -> float:
        g, h = dict(), dict() 

        # Initialize
        OPEN.add(start)
        Q.add(value=start, priority=0)
        MAX_BEAM_WIDTH = 10000
        WEIGHTING_FACTOR = 0.1
        g[start] = 0
        h[start] = 0

        while not Q.empty():

            # new set of nodes whose heuristic values must be calculated as a batch
            NEW = []

            # Get the top N nodes in OPEN with lowest f
            nodes = Q.popn(MAX_BEAM_WIDTH)
            print('OPEN:', len(OPEN), 'CLOSED:', len(CLOSED))
            print('Expanding {} nodes...'.format(len(nodes)))
            st = time.time()

            # Process this 'beam' of nodes
            for u, priority in nodes:
                hash_u = hash(u)

                # If we reached the target node
                if u == target:
                    return self.get_path(parent, start, target)

                # Consider all next nodes
                for a in self.get_next_actions(u):
                    v = u.apply_action(a)
                    hash_v = hash(v)

                    # If v is neither closed nor open
                    if v not in CLOSED and v not in OPEN:

                        # Open v and update it's cost and parent
                        OPEN.add(v)
                        NEW.append(v)
                        g[v] = g[u] + 1
                        parent[v] = (u, a)

                    # Else, if the current cost of getting to v
                    # is larger than going from u to v
                    elif g[v] > g[u] + 1:

                        # Update the cost and parent of v
                        g[v] = g[u] + 1
                        parent[v] = (u, a)

                        # If v closed, and hence not open
                        if v in CLOSED:

                            # Re-open v, and heuristic must be recalculated
                            CLOSED.remove(v)
                            OPEN.add(v)
                            NEW.append(v)

                # Finished exploring children of u,
                # so now close u
                OPEN.remove(u)
                CLOSED.add(u)

            en = time.time()
            print('finished expanding', en - st)

            st = time.time()
            # Calculate and update heuristics in a batch
            print('Calculating heuristic...')
            h = self.heuristic(NEW, target)
            for idx, v in enumerate(NEW):
                hash_v = hash(v)
                priority = WEIGHTING_FACTOR * g[v] + h[idx]
                # Note: v might already be in the priority queue,
                #       since we may have re-opened it.
                #       Thus, PriorityQueue must support updating weights.
                Q.add(value=v, priority=priority)
            en = time.time()
            print('finished heuristics', en - st)

        return None
