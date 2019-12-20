from abstract import State, Action, Graph

adj = {
    'a': ['b', 'c'],
    'b': ['a', 'd'],
    'c': ['d'],
    'd': ['a', 'e'],
    'e': [],
    'f': ['g'],
    'g': ['h'],
    'h': []
}


class ExampleAction(Action):

    def __init__(self, edge):
        self.edge = edge

    def __str__(self):
        return str(self.edge)

    def __repr__(self):
        return str(self.edge)

class ExampleState(State):

    def __init__(self, label):
        self.label = label

    def __hash__(self):
        return ord(self.label)

    def __eq__(self, other):
        return self.label == other.label

    def __lt__(self, other):
        return self.label < other.label

    def __str__(self):
        return str(self.label)

    def __repr__(self):
        return str(self.label)

class ExampleGraph(Graph):

    def __init__(self):
        pass

    def get_next_actions(self, s: State) -> list:
        return [ExampleAction((s.label, nxt)) for nxt in adj[s.label]]

    def apply_action(self, a: Action, s: State) -> State:
        if a.edge[0] != s.label:
            print('apply_action :: Action {} cannot be applied to state {}'.format(str(a), str(s)))
        return ExampleState(a.edge[1])

    def heuristic(self, s1: State, s2: State) -> float:
        return 0

graph = ExampleGraph()
start = ExampleState('a')
target = ExampleState('d')
print(graph.connected(start=start, target=target))
