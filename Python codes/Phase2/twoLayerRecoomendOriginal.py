__author__ = 'USER007'


import numpy
import math
import bintrees
import os
import sys


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
    rating_matrix = numpy.zeros((max(user_ids) + 1, max(item_ids) + 1), dtype=int)
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


def calc_user_similarity_from_rating_matrix(rating_mat_path):
    # Load rating matrix data and positive matrix data.
    rating_matrix = numpy.loadtxt(rating_mat_path, dtype=float, delimiter="\t")

    # Normalize rating data
    for i in xrange(1, rating_matrix.shape[0]):
        sys.stdout.write('\r')
        sys.stdout.write("Process of normalize:%f%%\t" % (round(1.0 * i / rating_matrix.shape[0], 4) * 100))
        sys.stdout.flush()

        r = rating_matrix[i, :]
        avg = r.sum() / len(r[r != 0])
        variance = numpy.zeros((1, rating_matrix.shape[1]), dtype=float)[0, :]
        variance[r != 0] = avg
        rating_matrix[i, :] = rating_matrix[i, :] - variance

    # Compute user similarities.
    user_sim_graph_data = numpy.zeros((rating_matrix.shape[0], rating_matrix.shape[0]))
    for i in xrange(1, rating_matrix.shape[0]):
        sys.stdout.write("\rProcess of user similarity computation:%f%%" % (1.0 * i / rating_matrix.shape[0] * 100))

        for j in xrange(1, rating_matrix.shape[0]):
            if i == j:
                user_sim_graph_data[i, j] = 0
            elif i < j:
                user_sim_graph_data[i, j] = round(cosine_similarity(rating_matrix[i], rating_matrix[j]), 2)
                user_sim_graph_data[j, i] = user_sim_graph_data[i, j]
    print 'Successful\t[Computing user similarities completed]'

    # Save user similarities to file.
    with open('../dataset/output/user_similarities.txt', 'w+') as f:
        for i in xrange(1, user_sim_graph_data.shape[0]):
            line = ""
            for j in xrange(1, user_sim_graph_data.shape[0]):
                line += user_sim_graph_data[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')


def calc_user_similarity_from_gene(normalized_user_gene_file_path):
    # Load raw user gene data.
    user_number = 0
    gene_feature_number = 0
    raw_data = []
    with open(normalized_user_gene_file_path) as f:
        for line in f:
            line = line.replace('\n', '')
            raw_data.append(line)
            user_number += 1
            gene_feature_number = len(line.split('\t'))
    f.close()
    print 'Success [Load user gene data complete]'

    # Format row user gene data into n by m numpy matrix.
    user_gene = numpy.zeros((user_number, gene_feature_number), dtype=float)
    for i in xrange(0, user_number):
        values = raw_data[i].split('\t')
        for j in xrange(0, gene_feature_number):
            user_gene[i, j] = values[j]
    print 'Success [Format item gene data complete]'

    # Drop the user_id feature data.
    user_gene = numpy.delete(user_gene, 0, 1)

    # Calculate user similarity by user' gene data.
    user_similarity = numpy.zeros((user_number, user_number), dtype=float)
    for i in xrange(0, user_number):
        print 'Similarity computing:\t' + str(round(1.0 * i / user_number * 100, 2))
        for j in xrange(0, user_number):
            if i == j:
                user_similarity[i, j] = 0
            elif i < j:
                user_similarity[i, j] = round(cosine_similarity(user_gene[i], user_gene[j]), 2)
                user_similarity[j, i] = user_similarity[i, j]
    print 'Success [Similarity computation task complete]'

    # Save similarity data to file.
    with open('../dataset/output/user_gene_similarity_data.txt', 'w+') as f:
        for i in xrange(0, user_number):
            line = ""
            for j in xrange(0, user_number):
                line += user_similarity[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')
    print 'Success [Saving data to file complete]'


def calc_item_similarity_from_gene(normalized_item_gene_file_path):
    # Load raw item gene data.
    item_number = 0
    gene_feature_number = 0
    row_data = []
    with open(normalized_item_gene_file_path) as f:
        for line in f:
            line = line.replace('\n', '')
            row_data.append(line)
            item_number += 1
            gene_feature_number = len(line.split('\t'))
    f.close()
    print 'Success [Load item gene data complete]'

    # Format row item gene data into n by m numpy matrix.
    item_gene = numpy.zeros((item_number, gene_feature_number), dtype=int)
    for i in xrange(0, item_number):
        values = row_data[i].split('\t')
        for j in xrange(0, gene_feature_number):
            item_gene[i, j] = values[j]
    print 'Success [Format item gene data complete]'

    # Calculate item similarity by items' gene data.
    item_similarity = numpy.zeros((item_number, item_number), dtype=float)
    for i in xrange(0, item_number):
        print 'Similarity computing:\t' + str(round(1.0 * i / item_number * 100, 2))
        for j in xrange(0, item_number):
            if i == j:
                item_similarity[i, j] = 0
            else:
                item_similarity[i, j] = round(cosine_similarity(item_gene[i], item_gene[j]), 2)
                item_similarity[j, i] = item_similarity[i, j]
    print 'Success [Similarity computation task complete]'

    # Save similarity data to file.
    with open('../dataset/output/item_gene_similarity_data.txt', 'w+') as f:
        for i in xrange(0, item_number):
            line = ""
            for j in xrange(0, item_number):
                line += item_similarity[i][j].__str__() + '\t'
            f.write(line[:-1] + '\n')
    print 'Success [Saving data to file complete]'


def cosine_similarity(v1, v2):
    numerator = sum(v1 * v2)
    denominator = math.sqrt(sum(numpy.power(v1, 2))) * math.sqrt(sum(numpy.power(v2, 2)))
    sim = 1.0 * numerator / denominator

    return sim


class SimilarUser:
    original_user_id = -1
    alike_user_id = -1
    similarity = -1

    def __init__(self, original_user_id, alike_user_id, similarity):
        self.original_user_id = original_user_id
        self.alike_user_id = alike_user_id
        self.similarity = similarity

    def __cmp__(self, other):
        return cmp(self.similarity, other.similarity)

    def __str__(self):
        return self.alike_user_id.__str__() + '\t' + self.similarity.__str__()


class SimilarItem:
    original_item_id = -1
    alike_item_id = -1
    similarity = -1

    def __init__(self, original_item_id, alike_item_id, similarity):
        self.original_item_id = original_item_id
        self.alike_item_id = alike_item_id
        self.similarity = similarity

    def __cmp__(self, other):
        return cmp(self.similarity, other.similarity)


class RatingData:
    user_id = -1
    item_id = -1
    rating = -1

    def __init__(self, user_id, item_id, rating):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating

    def __cmp__(self, other):
        return cmp(self.rating, other.rating)

    def __str__(self):
        return '[user][item][rating]:\t' + self.user_id.__str__() + '\t'\
               + self.item_id.__str__() + '\t' + self.rating.__str__()


class RecommendItem:
    recommend_to_user_id = 0
    recommend_from_user_id = 0
    user_similarity = 0

    item_id = 0
    original_rating = 0
    total_referenced_cnt = 0

    value = 0
    weight = 0

    def __init__(self, recommend_to_user_id, recommend_from_user_id, user_similarity,
                 item_id, original_rating):
        self.recommend_to_user_id = recommend_to_user_id
        self.recommend_from_user_id = recommend_from_user_id
        self.user_similarity = user_similarity
        self.item_id = item_id
        self.original_rating = original_rating
        self.total_referenced_cnt += 1

        self.value = original_rating * user_similarity
        self.weight = user_similarity

    def attach_another_connection(self, original_rating, user_similarity):
        self.value += original_rating * user_similarity
        self.weight += user_similarity
        self.total_referenced_cnt += 1

    def get_predict_rating(self):
        if self.total_referenced_cnt == 1:
            return round(self.value, 1)
        else:
            return round(1.0 * self.value / self.weight, 1)

    def __cmp__(self, other):
        return cmp(self.get_predict_rating(), other.get_predict_rating())

    def __str__(self):
        return 'Recommend item [' + str(self.item_id) + '] with predict rating at ['\
               + str(self.get_predict_rating()) + ']'


def make_recommendation(user_similarity_data_path, user_rating_data_path, user_id, recommend_num):
    # Load user similarity data and item similarity data.
    user_sim_matrix = numpy.loadtxt(user_similarity_data_path, dtype=float, delimiter="\t")
    user_rating_matrix = numpy.loadtxt(user_rating_data_path, dtype=float, delimiter="\t")

    # Drop the zero padding data(both zero row and zero column).
    user_rating_matrix = numpy.delete(user_rating_matrix, 0, 0)
    user_rating_matrix = numpy.delete(user_rating_matrix, 0, 1)
    print 'Success [Loading data completed]'

    # Find user's neighbors
    # user_id minus one because the matrix is zero-based index, but the user_id begins from one.
    user = user_sim_matrix[user_id - 1, :]
    top_similar_users = user.argsort()[-recommend_num:][::-1]
    neighbor_users = []
    for i in xrange(0, len(top_similar_users)):
        if user[top_similar_users[i]] > 0:
            neighbor_users.append(SimilarUser(user_id, top_similar_users[i] + 1, user[top_similar_users[i]]))
    print 'Success [Finding user neighbors completed]'

    # Find those items that the given user has been rated.
    # i plus one because the matrix is zero-based index, but the item_id begins from one.
    rated_items = []
    for i in xrange(0, user_rating_matrix.shape[1]):
        if user_rating_matrix[user_id - 1, i] != 0:
            rated_items.append(RatingData(user_id, i + 1, user_rating_matrix[user_id - 1, i]))

    # Sort rated_items so that I can find the given user's most like items.
    rated_items.sort(reverse=True)
    print 'Success [Finding rated items completed]'

    # Compute recommendations.
    recommendations = bintrees.RBTree()
    for u in neighbor_users:
        for i in xrange(0, user_rating_matrix.shape[1]):
            if user_rating_matrix[u.alike_user_id - 1, i] != 0:
                if recommendations.__contains__(i + 1):
                    item = recommendations.get(i + 1)
                    recommendations.remove(i + 1)
                    item.attach_another_connection(user_rating_matrix[u.alike_user_id - 1, i],
                                                   user_sim_matrix[user_id - 1, u.alike_user_id - 1])
                    recommendations.insert(i + 1, item)
                else:
                    item = RecommendItem(user_id, u.alike_user_id,
                                         user_sim_matrix[user_id - 1, u.alike_user_id - 1],
                                         i + 1, user_rating_matrix[u.alike_user_id - 1, i])
                    recommendations.insert(i + 1, item)

    # Remove those items that the given user has been rated.
    candidates = []
    for key in recommendations.keys():
        already_rated = False
        for item in rated_items:
            if item.item_id == key:
                already_rated = True
                break
        if not already_rated:
            candidates.append(recommendations.get(key))
    candidates.sort(reverse=True)

    # Save recommended items to file.
    cnt = 0
    with open('../dataset/output/recommendation_to_user_' + str(user_id) + '.txt', 'w+') as f:
        for i in candidates:
            if cnt >= recommend_num:
                break
            line = str(user_id) + '\t' + str(i.item_id) + '\t' + str(i.get_predict_rating())
            f.write(line + '\n')
            cnt += 1
        f.close()


def validate_prediction(test_bench_path, prediction_item_path, user):
    # Load test bench data and predict data.
    test_bench_data = numpy.loadtxt(test_bench_path, dtype=int, delimiter="\t")
    prediction_data = numpy.loadtxt(prediction_item_path, dtype=float, delimiter="\t")

    # Get the given user's test bench data
    test_bench = test_bench_data[test_bench_data[:, 0] == user, :]
    test_bench = test_bench[:, [0, 1, 2]]

    prediction = bintrees.RBTree()
    for i in range(0, len(prediction_data)):
        if not prediction.__contains__(prediction_data[i, 1]):
            prediction.insert(prediction_data[i, 1], prediction_data[i, 2])

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(0, len(test_bench)):
        real_value = True if test_bench[i, 2] >= 4 else False

        if prediction.__contains__(test_bench[i, 1]):
            predict_value = True if prediction.get(test_bench[i, 1]) >= 4 else False
        else:
            predict_value = False

        if real_value == predict_value:
            if real_value is True:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if real_value is True and predict_value is False:
                false_negative += 1
            else:
                false_positive += 1

    print 'True positive\t' + str(true_positive)
    print 'False positive\t' + str(false_positive)
    print 'True negative\t' + str(true_negative)
    print 'False negative\t' + str(false_negative)


def automatically_recommend(user_id, recommend_num):
    if not os.path.isfile('../dataset/output/user_rating_data.txt'):
        if not os.path.isfile('../dataset/input/u1_BIG.base'):
            print 'Error\t[u1_BIG.base not found]'
            return
        else:
            convert_to_rating_mat('../dataset/input/u1_BIG.base')
    print 'Success\t[Got rating data]'

    if not os.path.isfile('../dataset/output/user_similarities.txt'):
        if not os.path.isfile('../dataset/output/user_rating_data.txt'):
            print 'Error\t[user_rating_data.txt not found]'
            return
        else:
            calc_user_similarity_from_rating_matrix('../dataset/output/user_rating_data.txt')
    print 'Success\t[Got user similarity data]'

    if not os.path.isfile('../dataset/output/recommendation_to_user_' + str(user_id) + '.txt'):
        if not os.path.isfile('../dataset/output/user_similarities.txt'):
            print 'Error\t[user_similarities.txt not found]'
            return
        else:
            make_recommendation('../dataset/output/user_similarities.txt', '../dataset/output/user_rating_data.txt',
                                user_id, recommend_num)
    print 'Success\t[Got recommendation data]'

    validate_prediction('../dataset/input/u1_BIG.test',
                        '../dataset/output/recommendation_to_user_' + str(user_id) + '.txt', user_id)


if __name__ == '__main__':
    # normalized_item_gene_data_path = '../dataset/output/normalized_item_genes_data.txt'
    # calc_item_similarity_from_gene(normalized_item_gene_data_path)

    # normalized_user_gene_data_path = '../dataset/output/normalized_user_genes_data.txt'
    # calc_user_similarity_from_gene(normalized_user_gene_data_path)

    # user_gene_similarity_data = '../dataset/output/user_gene_similarity_data.txt'
    # user_rating_data = '../dataset/output/user_rating_data.txt'
    # make_recommendation(user_gene_similarity_data, user_rating_data, 1, 1000)

    # validate_prediction('../dataset/input/u1_BIG.test', '../dataset/output/recommendation_to_user_1.txt', 1)
    automatically_recommend(1, 100)
