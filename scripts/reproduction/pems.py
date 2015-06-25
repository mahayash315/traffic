import io
import string
import numpy as np

def load_pems(filename):
    dataset = None
    with open(filename, 'U') as fh:
        line = fh.readline()
        while line != "":
            rows = line.strip()[1:-1].split(';')
            if (dataset == None):
                dataset = [[] for _ in xrange(len(rows))]
            for i, row in enumerate(rows):
                data = []
                for col in row.split(' '):
                    data.append(string.atof(col))
                dataset[i].extend(data)
            line = fh.readline()
    return np.asarray(dataset, dtype=np.float32)

if __name__ == '__main__':
    dataset = load_pems('../../data/PEMS-SF/PEMS_train')