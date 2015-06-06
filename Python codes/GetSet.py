__author__ = 'USER007'

def read_data(path):
    _ids = []
    with open(path) as f:
        for line in f:
            line = line.replace('\n', '')
            values = line.split(' ')

            if len(values) > 0:
                for v in values:
                    #print 'has %s '% v
                    _ids.append(int(v))
    f.close()

    l = list(set(_ids))
    l.sort()
    for v in l:
        print v

if __name__ == "__main__":
    read_data('top_rating_items.txt')