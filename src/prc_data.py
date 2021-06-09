from collections import Counter
from utils import *

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    Arguments:
        The text of tv scripts split into words
    
    Returns:
        A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    
    word_cnt = Counter(text)
    sorted_words = sorted(word_cnt, key=word_cnt.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_words)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}    
    
    return (vocab_to_int, int_to_vocab)

def token_lookup():
    
    """
    Generate a dict to turn punctuation into a token.
    Returns: 
        Tokenized dictionary where the key is the punctuation and the value is the token
    """

    token = {'.': '||period||',
             ',': '||comma||',
             '"': '||quotation_mark||',
             ';': '||semicolon||',
             '!': '||exclamation_mark||',
             '?': '||question_mark||',
             '(': '||left_parentheses||',
             ')': '||right_parentheses||',
             '-': '||dash||',
             '\n':'||return||'}

    return token

if __name__ == '__main__':

    data_dir = '../data/got_scripts.txt'
    preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)