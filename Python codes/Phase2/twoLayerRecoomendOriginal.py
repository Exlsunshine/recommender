__author__ = 'USER007'


import numpy
import math


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
        return '[user][item][rating]:\t' + self.user_id.__str__() + '\t' + self.item_id.__str__() + '\t' + self.rating.__str__()


def make_recommendation(user_gene_similarity_data_path, item_gene_similarity_data_path, normalized_user_rating_data_path, user_id, recommend_num):
    # Load user similarity data and item similarity data.
    user_sim_matrix = numpy.loadtxt(user_gene_similarity_data_path, dtype=float, delimiter="\t")
    item_sim_matrix = numpy.loadtxt(item_gene_similarity_data_path, dtype=float, delimiter="\t")
    normalize_user_rating_matrix = numpy.loadtxt(normalized_user_rating_data_path, dtype=float, delimiter="\t")
    print 'Success [Loading data completed]'

    # Find user's neighbors
    # user_id minus one because the matrix is zero-based index, but the user_id begins from one.
    user = user_sim_matrix[user_id - 1,:]
    top_similar_users = user.argsort()[-recommend_num:][::-1]
    neighbor_users = []
    for i in xrange(0, len(top_similar_users)):
        neighbor_users.append(SimilarUser(user_id, top_similar_users[i] + 1, user[top_similar_users[i]]))
    print 'Success [Finding user neighbors completed]'

    # Find those items that the given user has been rated.
    # i plus one because the matrix is zero-based index, but the item_id begins from one.
    rated_items = []
    for i in xrange(0, normalize_user_rating_matrix.shape[1]):
        if (normalize_user_rating_matrix[user_id - 1, i] != 0):
            rated_items.append(RatingData(user_id, i + 1, normalize_user_rating_matrix[user_id -1, i]))

    # Sort rated_items so that I can find the given user's most like items.
    rated_items.sort(reverse=True)
    print 'Success [Finding rated items completed]'



if __name__ == '__main__':
    # normalized_item_gene_data_path = '../dataset/output/normalized_item_genes_data.txt'
    # calc_item_similarity_from_gene(normalized_item_gene_data_path)

    # normalized_user_gene_data_path = '../dataset/output/normalized_user_genes_data.txt'
    # calc_user_similarity_from_gene(normalized_user_gene_data_path)

    user_gene_similarity_data = '../dataset/output/user_gene_similarity_data.txt'
    item_gene_similarity_data = '../dataset/output/item_gene_similarity_data.txt'
    normalized_user_rating_data = '../dataset/output/normalized_user_rating_data.txt'

    make_recommendation(user_gene_similarity_data, item_gene_similarity_data, normalized_user_rating_data,1, 20)
