__author__ = 'USER007'

"""
Convert u.item file to [gene1, gene2, gene3, gene4... gen19] format
Example result: 0   1   0   1   1   1   0
"""


if __name__ == '__main__':
    path = 'D:\Data\Graduate_1_Spring\Recommender System\Dataset\ml-100k\u.item'

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

    with open('./item_genes_data.txt', 'w+') as f:
        for i in genes:
            f.write(i[:-1] + '\n')
