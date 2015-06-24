__author__ = 'USER007'


import numpy
import bintrees
import heapq
import sys
import  Combination


class RatingData:
    user_id = -1
    item_id = -1
    rating = -1
    normalized_rating = -1

    def __init__(self, user_id, item_id, rating, normalized_rating):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.normalized_rating = normalized_rating

    def __cmp__(self, other):
        return cmp(self.rating, other.rating)

    def __str__(self):
        return '[user][item][rating]:\t' + self.user_id.__str__() + '\t'\
               + self.item_id.__str__() + '\t' + self.rating.__str__()


'''
Get all permutations from a given array.
'''


def perm(original_data, from_index, n, permutations):
    if from_index == n - 1:
        permutations.append(list(original_data))
        return

    for k in xrange(from_index, n):
        swap(original_data, from_index, k)
        perm(original_data, from_index + 1, n, permutations)
        swap(original_data, from_index, k)


'''
Swap the index i data with the index j data.
'''


def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp


def brute_force_markov_recommendation(rating_matrix_path, positive_matrix_path, positive_items_path, user_id, markov_depth, k):
    # Load rating matrix data, positive matrix data and positive rated items data.
    rating_matrix = numpy.loadtxt(rating_matrix_path, dtype=int, delimiter="\t")
    positive_matrix = numpy.loadtxt(positive_matrix_path, dtype=int, delimiter="\t")
    positive_items = numpy.loadtxt(positive_items_path, dtype=int, delimiter="\t")
    print 'Success\t[Loading data from file completed]'

    # Compute combinations and permutations.
    original_data = numpy.array(positive_items[:, 1])
    combinations = Combination.combination(original_data, markov_depth)
    permutations = []
    for comb in combinations:
        perm(comb, 0, len(comb), permutations)
    print 'Success\t[Computing permutations completed]'

    # Get those items that the given user has been rated.
    rated_items = bintrees.RBTree()
    for i in xrange(1, rating_matrix.shape[1]):
        if rating_matrix[user_id, i] != 0:
            rated_items.insert(i, i)
    print 'Success\t[Building RBTree completed]'

    top_k_recommendation = []
    min_probability = 0
    summation_cache = bintrees.RBTree()
    for i in xrange(0, len(permutations)):
        sys.stdout.write("Download progress: %f%%   \r" % (round(1.0 * i / len(permutations), 4) * 100))
        # + str(i) + '/\t' + str(len(permutations))
        recommendation = evaluate_possibilities(positive_matrix, permutations[i], rated_items, summation_cache)
        if len(recommendation) > 0:
            # Save all recommendations first.
            for r in recommendation:
                if r.probability > min_probability:
                    heapq.heappush(top_k_recommendation, r)

            # Only remain top k recommendations.
            while top_k_recommendation.__len__() > k:
                min_item = heapq.heappop(top_k_recommendation)
                min_probability = min_item.probability

    print top_k_recommendation.__len__()
    rcm = []
    for i in xrange(0, k):
        item = heapq.heappop(top_k_recommendation)
        print item
        rcm.append(item.item_id)

    validation('../dataset/input/u1_BIG.test', user_id, rcm)


def validation(test_bench_path, user_id, recommendations):
    test_bench_data = numpy.loadtxt(test_bench_path, dtype=int, delimiter="\t")

    # Get the given user's test bench data
    # test_bench format looks like:
    # user_id   item_id     rating_value
    test_bench = test_bench_data[test_bench_data[:, 0] == user_id, :]
    test_bench = test_bench[:, [0, 1, 2]]

    prediction = bintrees.RBTree()
    for i in range(0, len(recommendations)):
        if not prediction.__contains__(recommendations[i]):
            prediction.insert(recommendations[i], recommendations[i])

    right_cnt = 0
    false_cnt = 0
    miss_cnt = 0

    for i in range(0, len(test_bench)):
        if prediction.__contains__(test_bench[i, 1]):
            if test_bench[i, 2] >= 4:
                right_cnt += 1
            else:
                false_cnt += 1
        else:
            miss_cnt += 1

    print 'Right\t' + str(right_cnt)
    print 'False\t' + str(false_cnt)
    print 'Miss\t' + str(miss_cnt)

    # Save positive items information to file.
    with open('../dataset/output/report' + str(user_id) + '.txt', 'w+') as f:
        f.write('#Right\t' + str(right_cnt) + '\n')
        f.write('#False\t' + str(false_cnt) + '\n')
        f.write('#Miss\t' + str(miss_cnt) + '\n')

        for i in recommendations:
            f.write(str(i) + '\n')
        f.close()
    print 'Success\t[Saving positive items data completed]'



def evaluate_possibilities(positive_matrix, permutation, rated_items, summation_cache):
    # Simulate markov chain to calculate the probability.
    probability = 1
    from_index = 0
    to_index = from_index + 1

    while True:
        if to_index == len(permutation):
            break
        else:
            rating_cnt = positive_matrix[permutation[from_index], permutation[to_index]]
            if rating_cnt == 0:
                probability = 0
                break
            else:
                summation = 0
                if summation_cache.__contains__(permutation[from_index]):
                    summation = summation_cache.get(permutation[from_index])
                else:
                    summation = sum(positive_matrix[permutation[from_index], :])
                    summation_cache.insert(permutation[from_index], summation)

                access_prob = 1.0 * rating_cnt / summation
                probability *= access_prob
                from_index = to_index
                to_index += 1

    # If the markov chain is broken, (the given permutation could not form a chain, somewhere in the matrix is zero)
    if probability == 0:
        return []
    else:
        # If the markov chain is intact,
        # then make some recommendations based on the probability.
        recommendation = []
        for i in xrange(1, positive_matrix.shape[1]):
            rating_cnt = positive_matrix[permutation[from_index], i]
            if rating_cnt != 0 and not rated_items.__contains__(i):
                summation = 0
                if summation_cache.__contains__(permutation[from_index]):
                    summation = summation_cache.get(permutation[from_index])
                else:
                    summation = sum(positive_matrix[permutation[from_index], :])
                    summation_cache.insert(permutation[from_index], summation)

                access_prob = 1.0 * rating_cnt / summation
                recommendation.append(RecommendItems(i, probability * access_prob, permutation))

        return recommendation


class RecommendItems:
    item_id = 0
    probability = 0
    permutation = []

    def __init__(self, item_id, probability, permutation):
        self.item_id = item_id
        self.probability = probability
        self.permutation = permutation

    def __cmp__(self, other):
        return cmp(self.probability, other.probability)

    def __str__(self):
        return 'Item [' + str(self.item_id) + ']\t' + 'with probability [' + str(self.probability) + ']'


def get_top_k_positive_items(user_rating_path, user_id, k):
    # Load rating matrix data.
    rating_matrix = numpy.loadtxt(user_rating_path, dtype=int, delimiter="\t")
    print 'Success\t[Loading data from file completed]'

    # Get the given user's positive related items.
    r = rating_matrix[user_id, :]
    avg = 1.0 * r.sum() / len(r[r != 0])
    sub_set = []
    for i in range(1, rating_matrix.shape[1]):
        if rating_matrix[user_id, i] > avg:
            sub_set.append(RatingData(user_id, i, rating_matrix[user_id, i], rating_matrix[user_id, i] - avg))
    print 'Success\t[Found ' + len(sub_set).__str__() + ' positive rated items]'

    # Sort the sub_set, so that I can easily get the most favorite k items respect to the given user.
    sub_set.sort(reverse=True)
    sub_set = sub_set[0: k]

    # Save positive items information to file.
    with open('../dataset/output/' + 'top_' + str(k) + '_positive_items_' + str(user_id) + '.txt', 'w+') as f:
        for i in sub_set:
            line = str(i.user_id) + '\t' + str(i.item_id) + '\t' + str(i.rating) + '\t'\
                + str(round(i.normalized_rating, 2))
            f.write(line + '\n')
        f.close()
    print 'Success\t[Saving positive items data completed]'


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("../dataset/output/logfile.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


if __name__ == '__main__':
    sys.stdout = Logger()

    rating_matrix_path = '../dataset/output/user_rating_data.txt'
    positive_matrix_path = '../dataset/output/PI_matrix_BIG.txt'
    positive_items_path = '../dataset/output/top_10_positive_items_1.txt'

    # get_top_k_positive_items(user_rating_path, 1, 10)
    brute_force_markov_recommendation(rating_matrix_path, positive_matrix_path, positive_items_path, 1, 7, 1000)
