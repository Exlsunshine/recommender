__author__ = 'USER007'

import numpy as np
import time
import bintrees

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
    with open('./ratring_dat_' + current_milli_time().__str__() + '.txt','w+') as f:
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
    rating_matrix = np.loadtxt(open(path,'rb'),dtype=int, delimiter="\t")
    rows = rating_matrix.shape[0]
    columns = rating_matrix.shape[1]

    # Build matrix for PI.
    pi = np.zeros((columns, columns))

    # Compute P set for every user.
    for u in range(1, rows):
        print 1.0 * u / rows * 100
        r = rating_matrix[u,:]
        avg = 1.0 * r.sum() / len(r[r != 0])

        sub_set = []
        for i in range(1, columns):
            if rating_matrix[u, i] > avg:
                sub_set.append(i)

        for i in range(0, len(sub_set)):
            for j in range(0, len(sub_set)):
                if sub_set[i] != sub_set[j] :
                    pi[sub_set[i],sub_set[j]] += 1

    # Save two dimensions rating data to file.
    with open('./PI_matrix_' + current_milli_time().__str__() + '.txt','w+') as f:
        for i in xrange(columns):
            line = ""
            for j in xrange(columns):
                line += (int(pi[i,j]).__str__()) + '\t'
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

def get_sub_positive_graph(rating_mat_path, positive_mat_path, user):
    # Load rating matrix data and positive matrix data.
    rating_matrix = np.loadtxt(open(rating_mat_path,'rb'),dtype=int, delimiter="\t")
    positive_matrix = np.loadtxt(open(positive_mat_path,'rb'),dtype=int, delimiter="\t")
    print 'Success\t[Loading data from file completed]'

    # Get the given user's positive related items.
    r = rating_matrix[user,:]
    avg = r.sum() / len(r[r != 0])
    sub_set = []
    for i in range(1, rating_matrix.shape[1]):
        if rating_matrix[user, i] > avg:
            sub_set.append(i)
    print 'Success\t[Find ' + len(sub_set).__str__() + ' rated items]'

    # Build the given user's sub-positive graph.
    sub_graph = np.zeros((len(sub_set), rating_matrix.shape[1]))
    for i in range(0, len(sub_set)):
        sub_graph[i,:] = positive_matrix[sub_set[i],:]

    # Pick up none zero edge from the sub-graph.
    candidates = []
    for i in range(0, len(sub_set)):
        from_id = sub_set[i]
        for to_id in range(1, rating_matrix.shape[1]):
            if positive_matrix[from_id, to_id] != 0 :
                candidates.append(Edge(from_id, to_id, positive_matrix[from_id, to_id]))

    print 'Success\t[Build sub-graph completed]'
    print sub_graph
    print '----------'

    print 'Success\t[Find ' + len(candidates).__str__() + ' candidate edges]'

    # Save edges information to file.
    with open('./Candidate_edges_' + current_milli_time().__str__() + '.txt','w+') as f:
        for i in candidates:
            line = str(i.from_id) + '\t' + str(i.to_id) + '\t' + str(i.weight)
            f.write(line + '\n')
        f.close()

def make_recommendation(rating_mat_path, candidates_mat_path, user, k):
    # Load rating matrix data and candidates matrix data.
    rating_matrix = np.loadtxt(open(rating_mat_path,'rb'),dtype=int, delimiter="\t")
    candidates_matrix = np.loadtxt(open(candidates_mat_path,'rb'),dtype=int, delimiter="\t")

    # Find all items which the given use has rated.
    rated_items = bintrees.RBTree()
    row = rating_matrix[user,:]
    for i in range(1, rating_matrix.shape[1]):
        if row[i] != 0:
            rated_items.insert(i,i)

    # Re-build edges,
    # this time, I remove those edges whose to_id has been already rated by the given user.
    candidates = []
    for i in xrange(candidates_matrix.shape[0]):
        from_id = candidates_matrix[i, 0]
        to_id = candidates_matrix[i, 1]
        weight = candidates_matrix[i, 2]

        if not rated_items.__contains__(to_id):
            candidates.append(Edge(from_id, to_id, weight))

    candidates.sort(reverse=True)
    print 'Candidate edges:'
    for i in candidates:
        print i

    for i in range(0, len(candidates)):
        if i >= k:
            break

        print 'Algorithm recommends user to checkout out item:\t' + str(candidates[i].to_id)

if __name__ == "__main__":
    #convert_to_rating_mat('./u1_BIG.base')
    #calc_positive_mat_from_rating_mat('./ratring_dat_BIG.txt')
    #get_sub_positive_graph('./ratring_dat_BIG.txt', './PI_matrix_BIG.txt', 1)
    make_recommendation('./ratring_dat_BIG.txt','./Candidate_edges_BIG.txt',1, 10)