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
            return Node(-1,-1)
        else:
            return self.heap[0]

    def push_item(self, item):
        min_item = self.get_min_item()
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
            else :
                current_item = heapq.heappop(self.heap)
                heapq.heappush(self.heap, item)
                self.features.pop(current_item.key)
                self.features[item.key] = item.value

    def printheap(self):
        for item in self.heap:
            print(item)

if __name__ == '__main__':
    top_k = TopK(k=2)
    print(top_k.heap)
    top_k.push_item(Node(5, 1))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(3, 1))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(7, 2))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(7, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(9, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(10, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(9, 4))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(9, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(10, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(10, 1))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
    top_k.push_item(Node(10, 2))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.printheap()
