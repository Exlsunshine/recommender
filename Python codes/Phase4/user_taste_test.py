__author__ = 'USER007'


import numpy
import bintrees
import sys
import math


def find_most_active_user(raw_rating_data_path):
    raw_rating_data = numpy.loadtxt(raw_rating_data_path, dtype=int, delimiter="\t")
    freq = numpy.bincount(raw_rating_data[:, 0])
    active_user_id = numpy.argmax(freq)

    print 'The most active user\'s id is ' + active_user_id.__str__()
    return active_user_id


class AggregateItems:
    MAX_NUM_OF_CLASS = 19

    begin_date = sys.maxint
    end_date = (-sys.maxint - 1)
    aggregate_num = 0
    aggregate_gene = numpy.zeros(MAX_NUM_OF_CLASS)

    def __init__(self):
        begin_date = sys.maxint
        end_date = (-sys.maxint - 1)
        self.aggregate_num = 0
        self.aggregate_gene = numpy.zeros(self.MAX_NUM_OF_CLASS)

    def append(self, gene, date):
        # Update gene counter
        mask = 1
        for i in xrange(0, self.MAX_NUM_OF_CLASS):
            mask = mask << i
            self.aggregate_gene[self.MAX_NUM_OF_CLASS - 1 - i] += 0 if ((gene & mask) == 0) else 1

        # Update statistic item counter
        self.aggregate_num += 1

        # Update min-max date info
        if date < self.begin_date:
            self.begin_date = date
        elif date > self.end_date:
            self.end_date = date

    def get_gene_avg(self):
        return self.aggregate_gene / self.aggregate_num * 1.0

    def __cmp__(self, other):
        return cmp(self.begin_date, other.begin_date)

    def __str__(self):
        return str(self.aggregate_gene) + "\t" + str(self.begin_date)


def aggregate_rating_data_by_time(raw_rating_data_path, item_gene_data_path, user_id, k):
    # Load raw user rating data.
    raw_rating_data = test_bench_data = numpy.loadtxt(raw_rating_data_path, dtype=int, delimiter="\t")
    # Get all user rating date respect to the given user.
    user_rating = raw_rating_data[raw_rating_data[:, 0] == user_id,:]
    # Sort by rating time.
    user_rating = user_rating[user_rating[:, 3].argsort()]

    gene_hash_table = get_item_gene_data(item_gene_data_path)
    aggregate_items = []
    if k >= user_rating.shape[0]:
        # If k >= the NO. of user_rating's row, then we don't need to perform any aggregation,
        # just return the raw data.
        for i in xrange(0, user_rating.shape[0]):
            item = AggregateItems()
            item.append(gene_hash_table.get(str(user_rating[i, 1])), user_rating[i, 3])
            aggregate_items.append(item)
    else:
        # Aggregate data for every k nearby rows.
        cnt = 0
        item = AggregateItems()
        for i in xrange(0, user_rating.shape[0]):
            if cnt == k:
                aggregate_items.append(item)
                cnt = 0
                item = AggregateItems()

            item.append(gene_hash_table.get(str(user_rating[i, 1])), user_rating[i, 3])
            cnt += 1

        if cnt != 0:
            aggregate_items.append(item)

    aggregate_items.sort(reverse=True)
    return aggregate_items


def get_item_gene_data(path):
    # Extract last 19 gene data columns.
    gene_hash_table = bintrees.RBTree()
    with open(path) as f:
        for line in f:
            line = line.replace('\n', '')
            values = line.split('|')

            bits = 0
            cnt = 0
            i = len(values) - 1
            while cnt < 19:
                bits = (int(values[i]) << cnt) | bits
                i -= 1
                cnt += 1
            gene_hash_table.insert(values[0], bits)

    f.close()
    return gene_hash_table


def get_distance(v1, v2):
    return math.sqrt(sum(numpy.power((v1 - v2),2)))


if __name__ == '__main__':
    k = 60
    raw_rating_data_path = '../dataset/input/u1_BIG.base'
    item_gene_data_path = '../dataset/input/u.item'
    active_user_id = find_most_active_user(raw_rating_data_path)
    aggregate_items = aggregate_rating_data_by_time(raw_rating_data_path, item_gene_data_path, active_user_id, k)

    for i in xrange(0, len(aggregate_items), 2):
        if i + 1 < len(aggregate_items):
            print str(i + 1) + '\t' + str(get_distance(aggregate_items[i].get_gene_avg(),
                                                       aggregate_items[i + 1].get_gene_avg()))