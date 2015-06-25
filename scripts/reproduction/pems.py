import io
import string
import numpy
import theano

def load_pems(filename, from_day=0, days=60):
    to_day = from_day+days
    dataset = None

    with open(filename, 'U') as fh:
        day = 0
        line = fh.readline()
        while line != "":
            if (from_day <= day and day < to_day):
                rows = line.strip()[1:-1].split(';')
                if (dataset == None):
                    dataset = [[] for _ in xrange(len(rows))]
                for i, row in enumerate(rows):
                    data = []
                    for col in row.split(' '):
                        data.append(string.atof(col))
                    dataset[i].extend(data)
            elif (to_day <= day):
                break

            line = fh.readline()
            day += 1

    return numpy.asarray(dataset, dtype=theano.config.floatX).transpose()

def load_data(filename, from_day=0, days=60, r=0, d=1):
    dataset = load_pems(filename, from_day, days)

    m = dataset.shape[1] # number of observation location
    n = dataset.shape[0] # number of observation samples per location

    dataset_x = [[dataset[(j-(i%(r+1)))][int(i/(r+1))] for j in xrange(r,n-d)] for i in xrange(m*(r+1))]
    dataset_y = [[dataset[j][i] for j in xrange(r+d,n)] for i in xrange(m)]

    return (numpy.asarray(dataset_x, dtype=theano.config.floatX), numpy.asarray(dataset_y, dtype=theano.config.floatX))

if __name__ == '__main__':
    dataset = load_data('../../data/PEMS-SF/PEMS_train',days=2,r=1)