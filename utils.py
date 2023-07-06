def get_data(path = None):
    if not path:
        path = 'names.txt'

    with open(path, 'r') as f:
        words = f.read().splitlines()
    return words

def statistics(words):
    print('Total number of words: {}'.format(len(words)))
    print('Total number of unique words: {}'.format(len(set(words))))
    print('Average word length: {}'.format(sum(len(word) for word in words) / len(words)))
    print('Maximum word length: {}'.format(max(len(word) for word in words)))
    print('Minimum word length: {}'.format(min(len(word) for word in words)))
