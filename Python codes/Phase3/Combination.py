__author__ = 'USER007'

import numpy


def combination(original_data, k):
    # Init binary array.
    binary_flag = numpy.zeros((1, len(original_data)))
    for i in xrange(0, k):
        binary_flag[0, i] = 1
    print 'Success\t[Init binary data completed.]'
    print binary_flag

    # Begin computing all combinations.
    combinations = []
    while not last_k_elements_all_ones(binary_flag, k):
        for i in xrange(0, binary_flag.shape[1]):
            if i + 1 < binary_flag.shape[1]:
                if binary_flag[0, i] == 1 and binary_flag[0, i + 1] == 0:
                    record_combination(original_data, binary_flag, combinations)
                    swap(binary_flag[0, :], i, i + 1)
                    shift_left(binary_flag, i)
                    break
    # Save the last combination
    record_combination(original_data, binary_flag, combinations)
    print 'Success\t[Init compute combinations completed.]'

    return combinations


def shift_left(binary_flag, i):
    # Get the first zero flag from the most left.
    left = 0
    for j in xrange(0, i):
        if binary_flag[0, j] == 0:
            left = j
            break

    j = i - 1
    while j >= 0 and j >= left:
        if binary_flag[0, j] == 1:
            swap(binary_flag[0, :], left, j)
            left += 1
        j -= 1


def record_combination(original_data, binary_flag, combinations):
    condition = binary_flag[0, :] != 0
    comb = numpy.extract(condition, original_data)
    combinations.append(comb)


def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp


def last_k_elements_all_ones(binary_flag, k):
    i = binary_flag.shape[1] - 1
    cnt = 0
    while cnt < k:
        if binary_flag[0, i] == 0:
            return False
        else:
            cnt += 1
            i -= 1

    return True


if __name__ == "__main__":
    # Unit test.
    original_data = numpy.array([1, 2, 3, 4, 5, 6])
    k = 2
    combinations = combination(original_data, k)

    for comb in combinations:
        print comb
