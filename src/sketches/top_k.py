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
        self.values = {}
        self.heap = []
        self.keys = []
        self.count = 0
        self.EPS = 1.05

    def is_key_available(self, key):
        return key in self.features

    def get_value_for_key(self, key):
        return self.features.get(key) if self.is_key_available(key) else 0

    def is_heap_full(self):
        return self.count >= self.k

    def get_min_item(self):
        return 0.0 if self.count < self.k else self.heap[0].key

    def push(self, item):
        key = item.key
        value = item.value
        abs_value = abs(value)
        if self.is_key_available(key):
            self.features[key] = value
            position = self.values[key]
            current = self.heap[position].key
            top = (abs_value >= current * self.EPS)
            bottom = (abs_value <= current / self.EPS)
            if self.count < self.k:
                current = abs_value
            elif top or bottom:
                current = abs_value
                self.heapify(position + 1, True)
        elif self.count < self.k:
            self.heap.append(Node(abs_value, self.count))
            self.keys.append(key)
            self.values[key] = self.count
            self.features[key] = value
            self.count += 1
            if self.count == self.k:
                for idx in range(self.k // 2, -1, -1):
                    self.heapify(idx)
        elif abs_value > (self.get_min_item() * self.EPS):
            self.insert(key, value)

    def heapify(self, idx, update=False):
        current = self.heap[idx - 1]
        if update and idx > 1:
            parent_index = idx // 2
            parent = self.heap[parent_index - 1]
            if current.key < parent.key:
                self.values[self.keys[current.value]] = parent_index - 1
                self.values[self.keys[parent.value]] = idx - 1
                temp = current
                self.heap[idx - 1] = parent
                self.heap[parent_index - 1] = temp
                # current, parent = parent, current
                self.heapify(parent_index, True)
        left_index = 2 * idx
        right_index = 2 * idx + 1
        if left_index <= self.k and right_index <= self.k:
            left = self.heap[left_index - 1]
            right = self.heap[right_index - 1]
            left_smallest = left.key <= right.key
            sc_idx = left_index if left_smallest else right_index
            sc = left if left_smallest else right
            if sc.key < current.key:
                self.values[self.keys[current.value]] = sc_idx - 1
                self.values[self.keys[sc.value]] = idx - 1
                temp = sc
                self.heap[sc_idx - 1] = current
                self.heap[idx - 1] = temp
                # sc, current = current, sc
                self.heapify(sc_idx, update)

    def insert(self, key, value):
        min_pos = self.heap[0].value
        min_key = self.keys[min_pos]
        self.values.pop(min_key)
        self.features.pop(min_key)
        self.keys[min_pos] = key
        self.values[key] = 0
        self.features[key] = value
        self.heap[0].key = abs(value)
        self.heapify(1)

    def print_heap(self):
        print("printing heap")
        for item in self.heap:
            key = self.keys[item.value]
            value = self.features[key]
            print(key, value)


if __name__ == '__main__':
    top_k = TopK(k=8)
    top_k.push(Node(5, 0.001))
    top_k.print_heap()
    top_k.push(Node(3, 0.03))
    top_k.print_heap()
    top_k.push(Node(7, 2))
    top_k.print_heap()
    top_k.push(Node(7, 3))
    top_k.print_heap()
    top_k.push(Node(9, 3))
    top_k.print_heap()
    top_k.push(Node(10, 3))
    top_k.print_heap()
    top_k.push(Node(9, 4))
    top_k.print_heap()
    top_k.push(Node(9, 3))
    top_k.print_heap()
    top_k.push(Node(10, 3))
    top_k.print_heap()
    top_k.push(Node(10, 1))
    top_k.print_heap()
    top_k.push(Node(10, 2))
    top_k.print_heap()
    print(top_k.features)
    print(top_k.get_min_item())
