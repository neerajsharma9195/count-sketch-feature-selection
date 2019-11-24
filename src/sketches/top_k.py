import heapq
from collections import OrderedDict


class TopK(object):
    def __init__(self, k):
        self.k = k
        self.features = {}
        self.heap = []

    def get_min_item(self):
        return self.heap[0]

    def push_item(self, item):
        min_item = self.get_min_item()
        if item[1] > min_item[1]:
            if item[0] in self.features.keys():
                index = self.heap.index((item[0], self.features[item[0]])) # is it O(1)
                self.heap[index] = self.heap[-1]
                heapq.heappop(self.heap)
                heapq.heapify(self.heap)
                heapq.heappush(self.heap, (item[0], item[1]))
            else:
               if self.heap


        print("before push {} num items {}".format(self.heap, self.num_items))
        print("item {}".format(item))
        if self.num_items <= self.k:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, item)
            self.num_items += 1
        else:
            cur_min = heapq.nsmallest(1, self.heap)
            print("current min {}".format(cur_min))
            if item[0] in self.heap_dict:
                self.h
            if cur_min[0][1] < item[1]:
                print("popping {}".format(heapq.heappop(self.heap)))
                heapq.heappush(self.heap, item)

    def comparator(self, item1, item2):
        if item1[1] > item2[1]:
            return item1[1]
        elif item1[1] == item2[1]:
            return item1[0] > item1[1]


if __name__ == '__main__':
    top_k = TopK(k=2)
    print(top_k.heap)
    top_k.push_item((5, 1))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.push_item((3, 1))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.push_item((7, 2))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.push_item((7, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.push_item((9, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.push_item((10, 3))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.push_item((9, 4))
    print("Get item {}".format(top_k.get_min_item()))
    top_k.push_item((9, 3))
    print("Get item {}".format(top_k.get_min_item()))
    print("final heap {}".format(top_k.heap))
