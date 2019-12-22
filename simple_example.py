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
        return '({}, {})'.format(self.edge[0], self.edge[1])

    def __repr__(self):
        return str(self)


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
        return str(self)

    def apply_action(self, a: ExampleAction) -> 'ExampleState':
        if a.edge[0] != self.label:
            print('apply_action :: ExampleAction {} cannot be applied to state {}'.format(str(a), str(self)))
        return ExampleState(a.edge[1])


class ExampleGraph(Graph):

    def __init__(self):
        pass

    def get_next_actions(self, s: State) -> list:
        return [ExampleAction((s.label, nxt)) for nxt in adj[s.label]]

    def heuristic(self, s1: State, s2: State) -> float:
        return 0


graph  = ExampleGraph()
start  = ExampleState('a')
target = ExampleState('a')
path = graph.connected(start, target)
if path is None:
    print('No path')
elif path == []:
    if start != target:
        print('path == [], but expected start == target')
    print('Already solved (start == target)')
else:
    for s, a in path:
        print(s,'->', a)
