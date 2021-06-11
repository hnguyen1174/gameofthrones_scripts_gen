import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import *
import numpy as np
import torch.nn as nn
from generate import *
import yaml

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

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        # Set variables for class
        self.output_size = output_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        
        # Define model layers
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, 
                            batch_first=True)        
        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)
        embeds = self.embed(nn_input)
        
        lstm_output, hidden = self.lstm(embeds, hidden)
        output = lstm_output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        output = output.view(batch_size, -1, self.output_size)
        output = output[:, -1]
        
        return output, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        weight = next(self.parameters()).data

        # Check for a GPU
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print('No GPU found. Please use a GPU to train your neural network.')
        
        # initialize hidden state with zero weights, and move to GPU if available
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

def forward_back_prop(rnn, optimizer, criterion, inputs, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    if(train_on_gpu):
        
        rnn = rnn.cuda()
        inputs, target = inputs.cuda(), target.cuda()
    
    hidden = tuple([each.data for each in hidden])
    
    optimizer.zero_grad()
    
    outputs, hidden = rnn(inputs, hidden)
    
    loss = criterion(outputs, target)
    
    loss.backward()    
    
    clip=5
    nn.utils.clip_grad_norm_(rnn.parameters(), clip)

    optimizer.step()
    
    return loss.item(), hidden

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn

if __name__ == '__main__':

    config = load_config('config.yaml')

    # Check for a GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')

    int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess(config['processed_data_dir'])

    # Sequence Length
    sequence_length = config['sequence_length']

    # Batch Size
    batch_size = config['batch_size']

    # data loader - do not change
    train_loader = batch_data(int_text, sequence_length, batch_size)

    # Number of Epochs
    num_epochs = config['num_epochs']

    # Learning Rate
    learning_rate = config['learning_rate']

    # Model parameters
    # Vocab size
    vocab_size = len(vocab_to_int)

    # Output size
    output_size = vocab_size

    # Embedding Dimension
    embedding_dim = config['embedding_dim']

    # Hidden Dimension
    hidden_dim = config['hidden_dim']

    # Number of RNN Layers
    n_layers = config['n_layers']

    # Show stats for every n number of batches
    show_every_n_batches = config['show_every_n_batches']

    # create model and move to gpu if available
    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    if train_on_gpu:
        rnn.cuda()

    # defining loss and optimization functions for training
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training the model
    trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

    # saving the trained model
    save_model('./save/{}'.format(config['model_name']), trained_rnn)
    print('Model Trained and Saved')

    if config['generate']:
        
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
            gen_length
            )
        print(generated_script)

        f =  open("../outputs/generated_script_3.txt", "w")
        f.write(generated_script)
        f.close()