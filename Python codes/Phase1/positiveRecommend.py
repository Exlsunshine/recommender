__author__ = 'USER007'

import numpy as np
import time
import bintrees
import math
import os

current_milli_time = lambda: int(round(time.time() * 1000))

"""
Convert [userID, itemID, ratingValue, timestamp] format to [two dimensions rating matrix] format
"""


def convert_to_rating_mat(path):
    # Get all user IDs and item IDs.
    user_ids = set()
    item_ids = set()
    with open(path) as f:
        for line in f:
            values = line.split('\t')
            user_ids.add(int(values[0]))
            item_ids.add(int(values[1]))
    f.close()

    # Load rating data.
    # Dimensions plus one because I want one-based index instead of zero-based index.
    # It's convenient that rating_matrix[i][j] stands for userI gives itemJ a rating,
    # rather than rating_matrix[i - 1][j - 1] stands for userI gives itemJ.
    rating_matrix = [[0 for i in xrange(max(item_ids) + 1)] for j in xrange(max(user_ids) + 1)]
    rows = len(rating_matrix)
    columns = len(rating_matrix[0])
    print 'file contains rows\t' + rows.__str__()
    print 'file contains columns\t' + columns.__str__()

    with open(path) as f:
        for line in f:
            values = line.split('\t')
            uid = values[0]
            iid = values[1]
            rating = values[2]
            rating_matrix[int(uid)][int(iid)] = int(rating)
    f.close()

    # Save two dimensions rating data to file.
    with open('../dataset/output/user_rating_data.txt', 'w+') as f:
        for i in xrange(rows):
            line = ""
            for j in xrange(columns):
                line += rating_matrix[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')

"""
Compute [positive related items] matrix from [two dimensions rating matrix] format matrix
"""


def calc_positive_mat_from_rating_mat(path):
    # Get rating data from file.
    rating_matrix = np.loadtxt(path, dtype=int, delimiter="\t")
    rows = rating_matrix.shape[0]
    columns = rating_matrix.shape[1]

    # Build matrix for PI.
    pi = np.zeros((columns, columns))

    # Compute P set for every user.
    for u in range(1, rows):
        print 1.0 * u / rows * 100
        r = rating_matrix[u, :]
        avg = 1.0 * r.sum() / len(r[r != 0])

        sub_set = []
        for i in range(1, columns):
            if rating_matrix[u, i] > avg:
                sub_set.append(i)

        for i in range(0, len(sub_set)):
            for j in range(0, len(sub_set)):
                if sub_set[i] != sub_set[j]:
                    pi[sub_set[i], sub_set[j]] += 1

    # Save two dimensions rating data to file.
    with open('../dataset/output/PI_matrix_BIG.txt', 'w+') as f:
        for i in xrange(columns):
            line = ""
            for j in xrange(columns):
                line += (int(pi[i, j]).__str__()) + '\t'
            f.write(line[:-1] + '\n')


class Edge:
    from_id = -1
    to_id = -1
    weight = -1

    def __init__(self, from_id, to_id, weight):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight

    def __str__(self):
        return self.from_id.__str__() + '\t->\t' + self.to_id.__str__() + '\t:\t' + self.weight.__str__()

    def __cmp__(self, other):
        return cmp(self.weight, other.weight)


class RecommendItem:
    item_id = -1
    frequency = 0
    value = 0

    def __init__(self, item_id):
        self.item_id = item_id
        self.frequency = 0
        self.value = 0

    def add_edge(self, rating, freq):
        self.frequency += freq
        self.value += rating * freq

    def get_predict_rating(self):
        return round(1.0 * self.value / self.frequency, 0)

    def get_accurate_predict_rating(self):
        return 1.0 * self.value / self.frequency

    def __cmp__(self, other):
        return cmp(self.frequency, other.frequency)


def get_sub_positive_graph(rating_mat_path, positive_mat_path, user):
    # Load rating matrix data and positive matrix data.
    rating_matrix = np.loadtxt(rating_mat_path, dtype=int, delimiter="\t")
    positive_matrix = np.loadtxt(positive_mat_path, dtype=int, delimiter="\t")
    print 'Success\t[Loading data from file completed]'

    # Get the given user's positive related items.
    r = rating_matrix[user, :]
    avg = r.sum() / len(r[r != 0])
    sub_set = []
    for i in range(1, rating_matrix.shape[1]):
        if rating_matrix[user, i] > avg:
            sub_set.append(i)
    print 'Success\t[Find ' + len(sub_set).__str__() + ' rated items]'

    # Build the given user's sub-positive graph.
    sub_graph = np.zeros((len(sub_set), rating_matrix.shape[1]))
    for i in range(0, len(sub_set)):
        sub_graph[i, :] = positive_matrix[sub_set[i], :]

    # Pick up none zero edge from the sub-graph.
    candidates = []
    for i in range(0, len(sub_set)):
        from_id = sub_set[i]
        for to_id in range(1, rating_matrix.shape[1]):
            if positive_matrix[from_id, to_id] != 0:
                candidates.append(Edge(from_id, to_id, positive_matrix[from_id, to_id]))

    print 'Success\t[Build sub-graph completed]'
    print sub_graph
    print '----------'

    print 'Success\t[Find ' + len(candidates).__str__() + ' candidate edges]'

    # Save edges information to file.
    with open('../dataset/output/Candidate_edges_BIG_' + str(user) + '.txt', 'w+') as f:
        for i in candidates:
            line = str(i.from_id) + '\t' + str(i.to_id) + '\t' + str(i.weight)
            f.write(line + '\n')
        f.close()


def make_recommendation(rating_mat_path, candidates_mat_path, user, k):
    # Load rating matrix data and candidates matrix data.
    rating_matrix = np.loadtxt(rating_mat_path, dtype=int, delimiter="\t")
    candidates_matrix = np.loadtxt(candidates_mat_path, dtype=int, delimiter="\t")

    # Find all items which the given user has already rated.
    rated_items = bintrees.RBTree()
    row = rating_matrix[user, :]
    for i in range(1, rating_matrix.shape[1]):
        if row[i] != 0:
            rated_items.insert(i, i)

    # Re-build edges,
    # this time, I remove those edges whose to_id has been already rated by the given user.
    candidates = []
    unrated_items = bintrees.RBTree()
    for i in xrange(candidates_matrix.shape[0]):
        from_id = candidates_matrix[i, 0]
        to_id = candidates_matrix[i, 1]
        weight = candidates_matrix[i, 2]

        if not rated_items.__contains__(to_id):
            if not unrated_items.__contains__(to_id):
                recommendation = RecommendItem(to_id)
                recommendation.add_edge(rating_matrix[user, from_id], weight)
                unrated_items.insert(to_id, recommendation)
            else:
                recommendation = unrated_items.get(to_id)
                recommendation.add_edge(rating_matrix[user, from_id], weight)
                unrated_items.remove(to_id)
                unrated_items.insert(to_id, recommendation)

    # Sort candidates so that I can easily recommend k items to the given user form
    # 0th row to kth row.
    for key in unrated_items.keys():
        candidates.append(unrated_items.get(key))
    candidates.sort(reverse=True)

    # Print recommended items.
    for i in range(0, len(candidates)):
        if i >= k:
            break
        print i.__str__() + '\t:\talgorithm recommends user to checkout out item:\t' + str(candidates[i].item_id) \
              + '\t' + str(candidates[i].get_predict_rating())

    # Save recommended items to file.
    cnt = 0
    with open('../dataset/output/recommended_items_BIG_' + str(user) + '.txt', 'w+') as f:
        for i in candidates:
            if cnt >= k:
                break
            line = str(user) + '\t' + str(i.item_id) + '\t' + str(int(i.get_predict_rating()))
            f.write(line + '\n')
            cnt += 1
        f.close()


def validate_prediction(test_bench_path, prediction_item_path, user):
    # Load test bench data and predict data.
    test_bench_data = np.loadtxt(test_bench_path, dtype=int, delimiter="\t")
    prediction_data = np.loadtxt(prediction_item_path, dtype=int, delimiter="\t")

    # Get the given user's test bench data
    test_bench = test_bench_data[test_bench_data[:, 0] == user, :]
    test_bench = test_bench[:, [0, 1, 2]]

    prediction = bintrees.RBTree()
    for i in range(0, len(prediction_data)):
        if not prediction.__contains__(prediction_data[i, 1]):
            prediction.insert(prediction_data[i, 1], prediction_data[i, 2])

    common_in_total = 0
    error = 0.0
    right_prediction_cnt = 0
    false_prediction_cnt = 0

    for i in range(0, len(test_bench)):
        if prediction.__contains__(test_bench[i, 1]):
            common_in_total += 1

            print str(test_bench[i, 2]) + '\t' + str(prediction.get(test_bench[i, 1])) + '\t'\
                  + str((test_bench[i, 2] - prediction.get(test_bench[i, 1])))

            error += math.pow((test_bench[i, 2] - prediction.get(test_bench[i, 1])), 2)




            like = True if test_bench[i, 2] >= 4 else False
            predict = True if prediction.get(test_bench[i, 1]) >= 4 else False
            if like == predict:
                right_prediction_cnt += 1
            else:
                false_prediction_cnt += 1
            # if test_bench[i, 2] + prediction.get(test_bench[i, 1]) >= 8 and \
            #                         test_bench[i, 2] + prediction.get(test_bench[i, 1]) <= 10:
            #     right_prediction_cnt += 1
            # else:
            #     false_prediction_cnt += 1

    print 'MSE is:\t' + str(math.sqrt(error))
    print 'Common in total:\t' + str(common_in_total)
    print 'Right rate:\t' + str(1.0 * right_prediction_cnt / common_in_total * 100) + '\t' + str(right_prediction_cnt)
    print 'False rate:\t' + str(1.0 * false_prediction_cnt / common_in_total * 100) + '\t' + str(false_prediction_cnt)


def automatically_recommend(user, k):
    if not os.path.isfile('../dataset/output/user_rating_data.txt'):
        if not os.path.isfile('../dataset/input/u1_BIG.base'):
            print 'Error\t[u1_BIG.base not found]'
            return
        else:
            convert_to_rating_mat('../dataset/input/u1_BIG.base')
    print 'Success\t[Got rating data]'

    if not os.path.isfile('../dataset/output/PI_matrix_BIG.txt'):
        if not os.path.isfile('../dataset/output/user_rating_data.txt'):
            print 'Error\t[user_rating_data.txt not found]'
            return
        else:
            calc_positive_mat_from_rating_mat('../dataset/output/user_rating_data.txt')
    print 'Success\t[Got positive matrix data]'

    if not os.path.isfile('../dataset/output/Candidate_edges_BIG_' + str(user) + '.txt'):
        get_sub_positive_graph('../dataset/output/user_rating_data.txt', '../dataset/output/PI_matrix_BIG.txt', user)

    if not os.path.isfile('../dataset/output/recommended_items_BIG_' + str(user) + '.txt'):
        make_recommendation('../dataset/output/user_rating_data.txt',
                            '../dataset/output/Candidate_edges_BIG_' + str(user) + '.txt', user, k)

    validate_prediction('../dataset/input/u1_BIG.test',
                        '../dataset/output/recommended_items_BIG_' + str(user) + '.txt', user)

if __name__ == "__main__":
    # convert_to_rating_mat('./u1_BIG.base')
    # calc_positive_mat_from_rating_mat('./rating_dat_BIG.txt')
    # get_sub_positive_graph('./rating_dat_BIG.txt', './PI_matrix_BIG.txt', 1)
    # make_recommendation('./rating_dat_BIG.txt','./Candidate_edges_BIG_1.txt',1, 2000)
    # validate_prediction('./u1_BIG.test','./recommended_items_BIG_1.txt',1)
    automatically_recommend(1, 1000)
    # 1/42
