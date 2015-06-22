import math
import random
import time
import numpy as np
import plot as plt


class Generator(object):
    def __init__(self):
        self.last_state = 0

    def __iter__(self):
        return self

    def next(self):
        raise StopIteration

class SimpleGenerator(Generator):
    def __init__(self, num=1):
        super(SimpleGenerator, self).__init__()
        self.num = num
        self.i = 0

    def f(self, x):
        return math.sin(x / math.pi)

    def noise(self, x):
        return (random.random() * 0.1)

    def itrgenerate(self, x, l):
        y = self.f(x)
        y += self.noise(x)
        return y

    def generate(self, x):
        arr = [0 for _ in xrange(self.num)]
        for l in xrange(self.num):
            arr[l] = self.itrgenerate(x, l)
        return arr

    def next(self):
        x = self.i
        self.i += 1
        return self.generate(x)


if __name__ == '__main__':
    try:
        plotter = plt.Plotter()
        gen = SimpleGenerator()
        for y in gen:
            print("{}".format(y))
            plotter.append(y)
            time.sleep(0.5)
    except Exception as e:
        print(str(e))