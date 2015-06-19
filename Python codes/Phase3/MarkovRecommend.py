__author__ = 'USER007'


import numpy


class Edge:
    from_item_id = -1
    to_item_id = -1
    weight = -1

    def __init__(self, from_id, to_id, weight):
        self.from_item_id = from_id
        self.to_item_id = to_id
        self.weight = weight

    def __str__(self):
        return self.from_item_id.__str__() + '\t->\t' + self.to_item_id.__str__() + '\t:\t' + self.weight.__str__()

    def __cmp__(self, other):
        return cmp(self.weight, other.weight)


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


def brute_force_markov_recommendation():
    pass


def get_top_k_sub_positive_graph(user_rating_path, positive_matrix_path, user_id, k):
    # Load rating matrix data and positive matrix data.
    rating_matrix = numpy.loadtxt(user_rating_path, dtype=int, delimiter="\t")
    positive_matrix = numpy.loadtxt(positive_matrix_path, dtype=int, delimiter="\t")
    print 'Success\t[Loading data from file completed]'

    # Get the given user's positive related items.
    r = rating_matrix[user_id, :]
    avg = r.sum() / len(r[r != 0])
    sub_set = []
    for i in range(1, rating_matrix.shape[1]):
        if rating_matrix[user_id, i] > avg:
            sub_set.append(RatingData(user_id, i, rating_matrix[user_id, i], rating_matrix[user_id, i] - avg))
    print 'Success\t[Find ' + len(sub_set).__str__() + ' positive rated items]'

    # Sort the sub_set, so that I can easily get the most favorite k items respect to the given user.
    sub_set.sort(reverse=True)
    sub_set = sub_set[0: k]

    # Build the given user's sub-positive graph.
    # sub_graph drop the first column because the total items is N but the rating_matrix is (M + 1) by (N + 1)
    sub_graph = numpy.zeros((len(sub_set), rating_matrix.shape[1]))
    for i in range(0, len(sub_set)):
        sub_graph[i, :] = positive_matrix[sub_set[i].item_id, :]
    sub_graph = sub_graph[1:]
    print 'Success\t[Build sub-graph completed]'

    # Pick up none zero edge from the sub-graph.
    candidates = []
    for i in range(0, len(sub_graph)):
        from_id = sub_set[i].item_id
        for to_id in range(1, rating_matrix.shape[1]):
            if positive_matrix[from_id, to_id] != 0:
                candidates.append(Edge(from_id, to_id, positive_matrix[from_id, to_id]))

    # Save edges information to file.
    with open('../dataset/output/' + 'top_' + str(k) + '_candidate_edges_BIG_' + str(user_id) + '.txt', 'w+') as f:
        for i in candidates:
            line = str(i.from_item_id) + '\t' + str(i.to_item_id) + '\t' + str(i.weight)
            f.write(line + '\n')
        f.close()


if __name__ == '__main__':
    user_rating_path = '../dataset/output/user_rating_data.txt'
    positive_matrix_path = '../dataset/output/PI_matrix_BIG.txt'

    get_top_k_sub_positive_graph(user_rating_path, positive_matrix_path, 1, 10)
