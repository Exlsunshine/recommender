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
    with open('../dataset/output/user_rating_data.txt', 'w+') as f:
        for i in xrange(rows):
            line = ""
            for j in xrange(columns):
                line += rating_matrix[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')


def calc_user_similarity(rating_mat_path):
    # Load rating matrix data and positive matrix data.
    rating_matrix = np.loadtxt(rating_mat_path, dtype=float, delimiter="\t")

    # Normalize rating data
    for i in xrange(1, rating_matrix.shape[0]):
        print 1.0 * i / rating_matrix.shape[0] * 100
        r = rating_matrix[i, :]
        avg = r.sum() / len(r[r != 0])
        # rating_matrix[i,:] = r[r != 0] - avg
        variance = np.zeros((1, rating_matrix.shape[1]), dtype=float)[0, :]
        variance[r != 0] = avg
        rating_matrix[i, :] = rating_matrix[i, :] - variance

    # Save two digits to make the result more concise.
    rating_matrix.round(2)

    # Compute user similarities.
    user_sim_graph_data = np.zeros((rating_matrix.shape[0], rating_matrix.shape[0]))
    for i in xrange(1, rating_matrix.shape[0]):
        print 1.0 * i / rating_matrix.shape[0] * 100
        for j in xrange(i + 1, rating_matrix.shape[0]):
            user_sim_graph_data[i, j] = round(cosine_similarity(rating_matrix[i, :], rating_matrix[j, :]), 2)
            user_sim_graph_data[j, i] = user_sim_graph_data[i, j]
    print 'Successful\t[Complete computing user similarities]'

    # Save user similarities to file.
    with open('../dataset/output/user_similarities.txt', 'w+') as f:
        for i in xrange(0, user_sim_graph_data.shape[0]):
            line = ""
            for j in xrange(0, user_sim_graph_data.shape[0]):
                line += user_sim_graph_data[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')


def calc_vertical_normalize_data(rating_mat_path):
    # Load rating matrix data and positive matrix data.
    rating_matrix = np.loadtxt(rating_mat_path, dtype=float, delimiter="\t")

    # Normalize rating data
    for i in xrange(1, rating_matrix.shape[1]):
        print 1.0 * i / rating_matrix.shape[1] * 100
        c = rating_matrix[:, i]
        avg = c.sum() / len(c[c != 0])
        variance = np.zeros((1, rating_matrix.shape[0]), dtype=float)[0, :]
        variance[c != 0] = avg
        rating_matrix[:, i] = rating_matrix[:, i] - variance
    print 'Success [Complete normalizing rating data]'

    # Save two digits to make the result more concise.
    rating_matrix.round(2)

    # Save item similarities to file.
    with open('../dataset/output/vertical_normalized_user_rating_data.txt', 'w+') as f:
        for i in xrange(1, rating_matrix.shape[0]):
            line = ""
            for j in xrange(1, rating_matrix.shape[1]):
                line += rating_matrix[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')
    print 'Success [Complete saving normalized rating data]'


def calc_horizontal_normalize_data(rating_mat_path):
    # Load rating matrix data and positive matrix data.
    rating_matrix = np.loadtxt(rating_mat_path, dtype=float, delimiter="\t")

    # Normalize rating data
    for i in xrange(1, rating_matrix.shape[0]):
        print 1.0 * i / rating_matrix.shape[0] * 100
        r = rating_matrix[i,:]
        avg = r.sum() / len(r[r != 0])
        variance = np.zeros((1, rating_matrix.shape[1]), dtype=float)[0, :]
        variance[r != 0] = avg
        rating_matrix[i,:] = rating_matrix[i,:] - variance
    print 'Success [Complete normalizing rating data]'

    # Save two digits to make the result more concise.
    rating_matrix.round(2)

    # Save item similarities to file.
    with open('../dataset/output/horizontal_normalized_user_rating_data.txt', 'w+') as f:
        for i in xrange(1, rating_matrix.shape[0]):
            line = ""
            for j in xrange(1, rating_matrix.shape[1]):
                line += rating_matrix[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')
    print 'Success [Complete saving normalized rating data]'


def calc_item_similarity(rating_mat_path):
    # Load rating matrix data and positive matrix data.
    rating_matrix = np.loadtxt(rating_mat_path, dtype=float, delimiter="\t")

    # Normalize rating data
    for i in xrange(1, rating_matrix.shape[1]):
        print 1.0 * i / rating_matrix.shape[1] * 100
        c = rating_matrix[:, i]
        avg = c.sum() / len(c[c != 0])
        variance = np.zeros((1, rating_matrix.shape[0]), dtype=float)[0, :]
        variance[c != 0] = avg
        rating_matrix[:, i] = rating_matrix[:, i] - variance

    # Save two digits to make the result more concise.
    rating_matrix.round(2)

    # Compute item similarities.
    item_sim_graph_data = np.zeros((rating_matrix.shape[1], rating_matrix.shape[1]))
    for i in xrange(1, rating_matrix.shape[1]):
        print 1.0 * i / rating_matrix.shape[1] * 100
        for j in xrange(i + 1, rating_matrix.shape[1]):
            item_sim_graph_data[i, j] = round(cosine_similarity(rating_matrix[:, i], rating_matrix[:, j]), 2)
            item_sim_graph_data[j, i] = item_sim_graph_data[i, j]
    print 'Successful\t[Complete computing item similarities]'

    # Save item similarities to file.
    with open('./item_similarities.txt', 'w+') as f:
        for i in xrange(0, item_sim_graph_data.shape[0]):
            line = ""
            for j in xrange(0, item_sim_graph_data.shape[0]):
                line += item_sim_graph_data[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')


def find_common_items(v1, v2):
    common = (v1 != 0) & (v2 != 0)
    return common


def cosine_similarity(vector1, vector2):
    # Find common rating items.
    common = find_common_items(vector1, vector2)
    v1 = vector1[common]
    v2 = vector2[common]

    if (common == False).all():
        return 0

    # Calculate cosine similarity.
    numerator = sum(v1 * v2)
    denominator = math.sqrt(sum(np.power(vector1, 2))) * math.sqrt(sum(np.power(vector2, 2)))
    sim = 1.0 * numerator / denominator

    return sim


def pearson_similarity(vector1, vector2):
    # Compute average rating for each vector.
    avg1 = sum(vector1) / len(vector1)
    avg2 = sum(vector2) / len(vector2)

    # Find common rating items.
    common = find_common_items(vector1, vector2)
    v1 = vector1[common]
    v2 = vector2[common]

    # Calculate pearson correlation factor.
    numerator = sum((v1 - avg1) * (v2 - avg2))
    denominator = math.sqrt(sum(np.power(v1 - avg1, 2))) * math.sqrt((sum(np.power(v2 - avg2, 2))))
    sim = 1.0 * numerator / denominator

    return sim


if __name__ == '__main__':
    convert_to_rating_mat('../dataset/input/u1_BIG.base')
    calc_vertical_normalize_data('../dataset/output/user_rating_data.txt')
