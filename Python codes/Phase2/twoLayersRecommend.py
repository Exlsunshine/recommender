__author__ = 'USER007'

import numpy as np
import math

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
    rating_matrix = np.zeros((max(user_ids) + 1, max(item_ids) + 1), dtype=int)
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
    with open('./rating_dat_BIG.txt', 'w+') as f:
        for i in xrange(rows):
            line = ""
            for j in xrange(columns):
                line += rating_matrix[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')


def calc_user_similarity(rating_mat_path):
    # Load rating matrix data and positive matrix data.
    rating_matrix = np.loadtxt(rating_mat_path, dtype=int, delimiter="\t")

    # Normalize rating data
    for i in xrange(1, rating_matrix.shape[0]):
        print 1.0 * i / rating_matrix.shape[0] * 100
        r = rating_matrix[i, :]
        avg = r.sum() / len(r[r != 0])
        rating_matrix[i,:] = rating_matrix[i,:] - avg

    # Compute user similarities.
    user_sim_graph_data = np.zeros((rating_matrix.shape[0], rating_matrix.shape[0]))
    for i in xrange(1, rating_matrix.shape[0]):
        print 1.0 * i / rating_matrix.shape[0] * 100
        for j in xrange(i + 1, rating_matrix.shape[0]):
            user_sim_graph_data[i, j] = round(cosine_similarity(rating_matrix[i, :], rating_matrix[j, :]), 2)
            user_sim_graph_data[j, i] = user_sim_graph_data[i, j]
    print 'Successful\t[Complete computing user similarities]'

    # Save user similarities to file.
    with open('./user_similarities.txt', 'w+') as f:
        for i in xrange(1, user_sim_graph_data.shape[0]):
            line = ""
            for j in xrange(1, user_sim_graph_data.shape[0]):
                line += user_sim_graph_data[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')


def cosine_similarity(v1, v2):
    numerator = sum(v1 * v2)
    denominator = math.sqrt(sum(np.power(v1, 2))) * math.sqrt(sum(np.power(v2, 2)))
    sim = 1.0 * numerator / denominator

    return sim


def pearson_similarity(v1, v2):
    avg1 = sum(v1) / len(v1)
    avg2 = sum(v2) / len(v2)

    numerator = sum((v1 - avg1) * (v2 - avg2))
    denominator = math.sqrt(sum(np.power(v1 - avg1, 2)) * sum(np.power(v2 - avg2, 2)))
    sim = 1.0 * numerator / denominator

    return sim

if __name__ == '__main__':
    calc_user_similarity('./rating_dat_BIG.txt')

    # rating_matrix = np.zeros((3,5), dtype=int)
    # rating_matrix[1,0] = 1
    # rating_matrix[1,1] = 2
    # rating_matrix[1,2] = 3
    # rating_matrix[1,3] = 42
    # rating_matrix[1,4] = 5
    # rating_matrix[2,0] = 1
    # rating_matrix[2,1] = 2
    # rating_matrix[2,2] = 3
    # rating_matrix[2,3] = 4
    # rating_matrix[2,4] = 4
    # print rating_matrix
    #
    # v1 = np.array(rating_matrix[1,:])
    # v2 = np.array(rating_matrix[2,:])
    # print '--------'
    # print v1
    # print v2
    # print pearson_similarity(v1, v2)
    # print cosine_similarity(v1, v2)
