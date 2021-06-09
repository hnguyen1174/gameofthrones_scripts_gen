import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import *

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    
    features, targets = [], []
    
    for idxb in range(0, len(words)-sequence_length):
        idxe = idxb + sequence_length
        
        # Append features
        features.append(words[idxb:idxe])
        
        # Append targets
        targets.append(words[idxe])

    features = torch.from_numpy(np.asarray(features))
    targets = torch.from_numpy(np.asarray(targets))
    
    data = TensorDataset(features, targets)
    dataloader = DataLoader(data, shuffle=False, batch_size=batch_size)
    
    return dataloader

if __name__ == '__main__':

    # Check for a GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')

    int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()