import time

import numpy as np
import theano
from theano import tensor as T
import visualizer as plt

class TestBed:
    def __init__(self):
        self.window_size = 1
        self.batch_size = 1
        self.x = T.dvector('x') # input data
        self.y = T.dvector('y') # output data

