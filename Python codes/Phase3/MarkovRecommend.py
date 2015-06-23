__author__ = 'USER007'


import numpy
import bintrees
import heapq


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

    # Compute permutations.,
    # and get the first k columns.
    original_data = numpy.array(positive_items[:, 1])
    permutations = []
    original_data = original_data[0: markov_depth]
    perm(original_data, 0, len(original_data), permutations)
    print 'Success\t[Computing permutations completed]'

    # Get those items that the given user has been rated.
    rated_items = bintrees.RBTree()
    for i in xrange(1, rating_matrix.shape[1]):
        if rating_matrix[user_id, i] != 0:
            rated_items.insert(i, i)
    print 'Success\t[Building RBTree completed]'

    top_k_recommendation = []
    for i in xrange(0, len(permutations)):
        print str(round(1.0 * i / len(permutations), 2) * 100) + '%\t' + str(i) + '/\t' + str(len(permutations))
        recommendation = evaluate_possibilities(positive_matrix, permutations[i], rated_items)
        if len(recommendation) > 0:
            # Save all recommendations first.
            for r in recommendation:
                heapq.heappush(top_k_recommendation, r)

            # Only remain top k recommendations.
            while top_k_recommendation.__len__() > k:
                heapq.heappop(top_k_recommendation)

    print top_k_recommendation.__len__()
    for i in xrange(0, k):
        print heapq.heappop(top_k_recommendation)


def evaluate_possibilities(positive_matrix, permutation, rated_items):
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
                access_prob = 1.0 * rating_cnt / sum(positive_matrix[permutation[from_index], :])
                probability *= access_prob
                from_index = to_index
                to_index += 1

    # If the markov chain is broken, (the given permutation could not form a chain, somewhere in the matrix is zero)
    if probability == 0:
        return []
    else:
        # If the markov chain is intact,
        # then I make some recommendations based on the probability.
        recommendation = []
        for i in xrange(1, positive_matrix.shape[1]):
            rating_cnt = positive_matrix[permutation[from_index], i]
            if rating_cnt != 0 and not rated_items.__contains__(i):
                access_prob = 1.0 * rating_cnt / sum(positive_matrix[from_index, :])
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


if __name__ == '__main__':
    rating_matrix_path = '../dataset/output/user_rating_data.txt'
    positive_matrix_path = '../dataset/output/PI_matrix_BIG.txt'
    positive_items_path = '../dataset/output/top_10_positive_items_1.txt'

    # get_top_k_positive_items(user_rating_path, 1, 10)
    brute_force_markov_recommendation(rating_matrix_path, positive_matrix_path, positive_items_path, 1, 6, 20)
