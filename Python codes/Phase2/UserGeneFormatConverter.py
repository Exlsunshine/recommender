__author__ = 'USER007'

import numpy
import bintrees


"""
Load user occupation information.
"""


def load_occupation_data(occupation_data_path):
    occupation = bintrees.RBTree()
    index = 1
    with open(occupation_data_path) as f:
        for line in f:
            line = line.replace('\n', '')
            occupation.insert(line, index)
            index += 1

    return occupation


def load_user_gene_data(user_data_path, occupation):
    # Load raw user information data.
    user_raw_data = []
    with open(user_data_path) as f:
        for line in f:
            line = line.replace('\n', '')
            values = line.split('|')

            user_id = values[0]
            user_age = values[1]
            user_gender = values[2]
            user_occ = values[3]
            user_zipcode = values[4]

            row = [user_id, user_age, user_gender, user_occ, user_zipcode]
            user_raw_data.append(row)

    # Convert raw data to numerical data.
    # To drop zipcode information, I declare m * (n - 1) dimension matrix.(m is the number of user,
    # n is the number of features).
    user_data = numpy.zeros((len(user_raw_data), len(user_raw_data[0]) - 1))
    for i in xrange(0, len(user_raw_data)):
        values = user_raw_data[i]

        user_id = int(values[0])
        user_age = int(values[1])
        user_gender = 1 if values[2] == 'F' else 0
        user_occ = occupation.get(values[3])

        user_data[i] = [user_id, user_age, user_gender, user_occ]

    # Features scaling.
    user_data = std_dev_normalization(user_data)

    # Save scaled result to file.
    with open('../dataset/output/normalized_user_genes_data.txt', 'w+') as f:
        for i in xrange(0, len(user_data)):
            line = ""
            for j in xrange(0, len(user_data[0])):
                line += str(user_data[i, j]) + '\t'
            f.write(line[:-1] + '\n')

    return user_data


"""
x' = (x - min) / (max - min)
"""


def min_max_normalization(user_data):
    # Scale age and occupation feature value into [0, 1]
    user_age_min = min(user_data[:, 1])
    user_age_max_gap = max(user_data[:, 1]) - user_age_min
    user_occ_min = min(user_data[:, 3])
    user_occ_max_gap = max(user_data[:, 3]) - user_occ_min
    for i in xrange(0, len(user_data)):
        user_data[i, 1] = 1.0 * round((user_data[i, 1] - user_age_min) / user_age_max_gap, 2)
        user_data[i, 3] = 1.0 * round((user_data[i, 3] - user_occ_min) / user_occ_max_gap, 2)

    return user_data


"""
x' = (x - mean) / standard deviation(X)
"""


def std_dev_normalization(user_data):
    user_age = user_data[:, 1]
    user_gender = user_data[:, 2]
    user_occ = user_data[:, 3]

    user_age_mean = numpy.mean(user_age)
    user_age_std_dev = numpy.std(user_age)
    user_gender_mean = numpy.mean(user_gender)
    user_gender_std_dev = numpy.std(user_gender)
    user_occ_mean = numpy.mean(user_occ)
    user_occ_std_dev = numpy.std(user_occ)

    user_data[:, 1] = numpy.round((user_age - user_age_mean) / user_age_std_dev, 2)
    user_data[:, 2] = numpy.round((user_gender - user_gender_mean) / user_gender_std_dev, 2)
    user_data[:, 3] = numpy.round((user_occ - user_occ_mean) / user_occ_std_dev, 2)

    return user_data

if __name__ == '__main__':
    occupation_path = '../dataset/input/u.occupation'
    user_path = '../dataset/input/u.user'

    occupation = load_occupation_data(occupation_path)
    user = load_user_gene_data(user_path, occupation)
