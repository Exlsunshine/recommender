__author__ = 'USER007'

user_ids = []
item_ids = []
rating_matrix = [[]]

def format_data(path):
    _user_ids = set()
    _item_ids = set()
    with open(path) as f:
        for line in f:
            values = line.split('\t')
            _user_ids.add(values[0])
            _item_ids.add(values[1])
    f.close()

    for item in _user_ids:
        user_ids.append(int(item))
    for item in _item_ids:
        item_ids.append(int(item))
    user_ids.sort()
    item_ids.sort()
    rating_matrix = [[0 for i in xrange(max(item_ids) + 1)] for j in xrange(max(user_ids) + 1)]

    with open(path) as f:
        for line in f:
            values = line.split('\t')
            uid = values[0]
            iid = values[1]
            rating = values[2]
            rating_matrix[int(uid)][int(iid)] = rating
    f.close()

    with open('./ratring_dat.txt','w+') as f:
        for i in xrange(max(user_ids) + 1):
            all_zero = True
            line = ""
            for j in xrange(max(item_ids) + 1):
                line += rating_matrix[i][j].__str__() + '\t'
                if rating_matrix[i][j] != 0:
                    all_zero = False

            if not all_zero:
                f.write(line + '\n')

if __name__ == "__main__":
    format_data('D:\Data\Graduate_1_Spring\Recommender System\Dataset\ml-100k\u1.test')


