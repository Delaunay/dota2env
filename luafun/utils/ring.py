import array

# In the future we might want to discard weights that worsened the model
# and start from the best model so far
# to do so we can simply insert in order (ordered by reward)
# that way the worst model is discarded instead of the oldest
# import bisect 
# a = [1, 2, 4, 5] 
# bisect.insort(a, 3) 
# print(a)

class RingBuffer:
    """Simple ring buffer with constant memory usage, discard oldest value"""
    def __init__(self, size, dtype, default_val=0):
        if len(str(dtype)) > 1:
            self.array = [default_val] * size
        else:
            self.array = array.array(dtype, [default_val] * size)

        self.capacity = size
        self.offset = 0

    def __getitem__(self, item):
        if item < 0:
            item = item % self.offset
            
        if self.offset < self.capacity:
            return self.array[item]

        return self.array[(self.offset + item) % self.capacity]

    def __setitem__(self, item, value):
        if self.offset < self.capacity:
            self.array[item] = value

        end_idx = self.offset % self.capacity
        self.array[(end_idx + item) % self.capacity] = value

    def __iter__(self):
        return iter(self.to_list())

    def append(self, item):
        self.array[self.offset % self.capacity] = item
        self.offset += 1

    def to_list(self):
        if self.offset < self.capacity:
            return list(self.array[:self.offset])
        else:
            end_idx = self.offset % self.capacity
            return list(self.array[end_idx: self.capacity]) + list(self.array[0:end_idx])

    def __str__(self):
        return str(self.to_list())

    def __len__(self):
        return min(self.capacity, self.offset)

    def last(self):
        if self.offset == 0:
            return None

        return self.array[(self.offset - 1) % self.capacity]


if __name__ == '__main__':
    b = RingBuffer(10, None)

    r = 100

    for i in range(r):
        b.append(i)

    print(b)

    for i in range(r):
        b[i] = i + 1

    print(b)

    print(b[-1], b.last())

    print(a[-1])
    print(a[-2])

    print(a[0])
    print(a[1])
