from heapq import heappush, heappop

class PriorityQueue():
    """
    A simple PriorityQueue implementation, where values can be updated.
    ** This is definitely not thread safe **
    Somewhat inspired by https://docs.python.org/3.5/library/heapq.html#priority-queue-implementation-notes
    """

    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED_FLAG = 'X'

    def add(self, value, priority):
        """
        Add a new value or update the priority of an existing one.
        """
        if value in self.entry_finder:
            self.remove(value)
        entry = [priority, value]
        self.entry_finder[value] = entry
        heappush(self.pq, entry)

    def remove(self, value):
        """
        Mark a value as REMOVED.
        """
        entry = self.entry_finder.pop(value)
        entry[-1] = self.REMOVED_FLAG

    def pop(self):
        """
        Remove and return the lowest priority tasks.
        """
        while self.pq:
            priority, value = heappop(self.pq)
            if value is not self.REMOVED_FLAG:
                del self.entry_finder[value]
                return (value, priority)
        raise KeyError('pop from an empty priority queue')

    def popn(self, n):
        """
        Remove and return the n lowest priority tasks.
        If there are fewer than n elements in the queue,
        all elements are removed and returned.
        """
        values = []
        while self.pq:
            if len(values) == n:
                break
            priority, value = heappop(self.pq)
            if value is not self.REMOVED_FLAG:
                del self.entry_finder[value]
                values.append((value, priority))
        return values

    def empty(self):
        return len(self.pq) == 0


if __name__ == '__main__':
    print('Testing Heap class...')
    Q = PriorityQueue()
    Q.add(value=1, priority=3)
    Q.add(value=1, priority=2)
    v, p = Q.pop()
    assert p == 2, "Incorrect value p = {}".format(p)
    assert v == 1, "Incorrect value v = {}".format(v)
    Q.add(value=1, priority=2)
    Q.add(value=7, priority=1)
    Q.add(value=8, priority=3)
    Q.add(value=9, priority=4)
    values = Q.popn(3)
    assert values == [(7, 1), (1, 2), (8, 3)], "Incorrect values = {}".format(values)
    values = Q.popn(1)
    assert values == [(9, 4)], "Incorrect values = {}".format(values)
    values = Q.popn(1)
    assert values == [], "Incorrect values = {}".format(values)
    print('Tests Passed')
