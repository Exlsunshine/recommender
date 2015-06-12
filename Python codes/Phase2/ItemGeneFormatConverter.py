__author__ = 'USER007'

"""
Convert u.item file to [gene1, gene2, gene3, gene4... gen19] format
Example result: 0   1   0   1   1   1   0

Note that: u.item data does not need to be normalized, because all it's gene data are either 1 or 0
"""


if __name__ == '__main__':
    path = '../dataset/input/u.item'

    # Extract last 19 gene data columns.
    genes = []
    with open(path) as f:
        for line in f:
            line = line.replace('\n', '')
            values = line.split('|')

            row = ""
            cnt = 0
            i = len(values) - 1
            while cnt < 19:
                row = str(values[i]) + "\t" + row
                i -= 1
                cnt += 1
            genes.append(row)
    f.close()

    # Save item gene data to file.
    with open('../dataset/output/normalized_item_genes_data.txt', 'w+') as f:
        for i in genes:
            f.write(i[:-1] + '\n')
