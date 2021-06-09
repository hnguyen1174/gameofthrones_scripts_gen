import numpy as np
import os

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

if __name__ == '__main__':

    data_dir = '../data/got_scripts.txt'
    text = load_data(data_dir)

    view_line_range = (0, 10)

    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

    lines = text.split('\n')
    print('Number of lines: {}'.format(len(lines)))
    word_count_line = [len(line.split()) for line in lines]
    print('Average number of words in each line: {}'.format(np.average(word_count_line)))

    print()
    print('The lines {} to {}:'.format(*view_line_range))
    print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))