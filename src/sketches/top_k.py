import heapq
from collections import OrderedDict


class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return str("{} : {}".format(self.key, self.value))


class TopK(object):
    def __init__(self, k):
        self.k = k
        self.features = {}
        self.heap = []
        self.count = 0

    def get_min_item(self):
        if self.count == 0:
            return Node(-1, -1)
        else:
            return heapq.nsmallest(1, self.heap)[0]

    def push_item(self, item):
        min_item = self.get_min_item()
        if self.count < self.k:
            if self.is_item_present(item):
                self.update_item(item)
                self.features[item.key] = item.value
                heapq.heapify(self.heap)
            else:
                heapq.heappush(self.heap, item)
                self.count += 1
                self.features[item.key] = item.value
        elif item.value > min_item.value:
            if self.is_item_present(item):
                self.update_item(item)
                heapq.heapify(self.heap)  # this heapify will work or not ...
                self.features[item.key] = item.value
            else:
                current_item = heapq.heappop(self.heap)
                heapq.heappush(self.heap, item)
                self.features.pop(current_item.key)
                self.features[item.key] = item.value


        '''
        if self.count < self.k:
            heapq.heappush(self.heap, item)
            self.count += 1
            self.features[item.key] = item.value
        elif item.value > min_item.value:
            if item.key in self.features.keys():
                for element in self.heap:
                    if element.key == item.key:
                        element.value = item.value
                        break
                heapq.heapify(self.heap)  # this heapify will work or not ...
                self.features[item.key] = item.value
            else:
                current_item = heapq.heappop(self.heap)
                heapq.heappush(self.heap, item)
                self.features.pop(current_item.key)
                self.features[item.key] = item.value
        '''

    def print_heap(self):
        print("printing heap")
        for item in self.heap:
            print(item.key, item.value)

    def get_item(self, pos):
        item = self.features.get(pos)
        if item is None:
            return 0
        else:
            return item

    def is_item_present(self, item):
        if item.key in self.features.keys():
            return True
        return False

    def update_item(self, item):
        if self.is_item_present(item):
            for i in self.heap:
                if item.key == i.key and item.value > i.value:
                    i.value = item.value


if __name__ == '__main__':
    top_k = TopK(k=8)
    print(top_k.heap)
    top_k.push_item(Node(5, 0.001))
    top_k.print_heap()
    top_k.push_item(Node(3, 0.03))
    top_k.print_heap()
    top_k.push_item(Node(7, 2))
    top_k.print_heap()
    top_k.push_item(Node(7, 3))
    top_k.print_heap()
    top_k.push_item(Node(9, 3))
    top_k.print_heap()
    top_k.push_item(Node(10, 3))
    top_k.print_heap()
    top_k.push_item(Node(9, 4))
    top_k.print_heap()
    top_k.push_item(Node(9, 3))
    top_k.print_heap()
    top_k.push_item(Node(10, 3))
    top_k.print_heap()
    top_k.push_item(Node(10, 1))
    top_k.print_heap()
    top_k.push_item(Node(10, 2))
    top_k.print_heap()
