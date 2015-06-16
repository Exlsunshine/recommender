__author__ = 'USER007'


import numpy
import datetime


def unix_timestamp_2_date(timestamp):
    date = []
    for i in xrange(0, len(timestamp)):
        date.append(datetime.datetime.fromtimestamp(timestamp[i]).strftime('%Y-%m-%d %H:%M:%S'))

    return date


if __name__ == '__main__':
    # Load data.
    data = numpy.loadtxt('D:\Data\Graduate_1_Spring\Recommender System\Dataset\ml-100k\u1.base',dtype=int,delimiter='\t')

    # Convert to date format from unix timestamp format.
    date = unix_timestamp_2_date(data[:,3])

    # Save date format to file.
    with open('../dataset/output/u1.dateformat.base', 'w+') as f:
        for i in xrange(0, data.shape[0]):
            line = ""
            for j in xrange(0, data.shape[1] - 1):
                line += data[i][j].__str__() + '\t'
            line += date[i]
            f.write(line + '\n')
