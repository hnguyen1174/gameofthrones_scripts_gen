import torch
from utils import *
import torch.nn.functional as F
from build_models import *

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, 
             sequence_length = 20, predict_len=1000, train_on_gpu=True):

    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        # top_i = np.where(top_i >= len(int_to_vocab)-1, len(int_to_vocab)-1, top_i)

        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())

        print('=========================')
        print(word_i)
        print('=========================')
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     

        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences

if __name__ == '__main__':

    # Check for a GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')

    _, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
    trained_rnn = load_model('./save/trained_rnn2')

    sequence_length = 10
    gen_length = 100
    prime_word = 'lannisters'

    pad_word = SPECIAL_WORDS['PADDING']
    generated_script = generate(
        trained_rnn, 
        vocab_to_int[prime_word], 
        int_to_vocab, 
        token_dict, 
        vocab_to_int[pad_word], 
        # gen_length
        )
    print(generated_script)

    f =  open("../outputs/generated_script_2.txt", "w")
    f.write(generated_script)
    f.close()