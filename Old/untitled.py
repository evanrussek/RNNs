# set up packages
import json
import numpy as np
import matplotlib.pyplot as plt
from res.sequential_tasks import pad_sequences, to_categorical
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# load in data files... 
sim_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'
human_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/human_trials.json'

def load_data(sim_data_path, human_data_path, n_train=1e5, n_test=1e5):

    train_file_idxs = range(1,16)
    test_file_idxs = range(16,31)

    train_files = [os.path.join(sim_data_path, str(i) + '.json') for i in train_file_idxs]
    test_files = [os.path.join(sim_data_path, str(i) + '.json') for i in test_file_idxs]

    a = [json.load(open(train_files[i])) for i in range(15)]
    train_trials = [item for sublist in a for item in sublist]
    del a
    train_data_sim = train_trials[:int(1e6)]

    test_trials = json.load(open(test_files[0]))
    test_data_sim = test_trials[:int(1e5)]

    human_data = json.load(open(human_data_path))
    
    return train_data_sim, test_data_sim, human_data

# train_data_sim, test_data_sim, human_data = load_data(sim_data_path, human_data_path)
# this is for fixations and choice... 
# this is for fixations and choice... 
def gen_batch_data_fixations_choice(batch_size, batch_idx, data, human_data=False):

    """
    Create sequence and target data for a batch

    Input: 
        batch_size: number of trials to include in batch
        batch_idx: index of data
        data: list of dicts, where each dict has 'values', 'fixations', and 'choice'
        human_data: this is just coded differently, so need to specify

    Returns:
        a tuple, (batch_data, batch_targets)
        batch_data is 3d array: batch_size x sequence_size x one-hot categorical encoding (3 here)
        batch_targets is 2d array: 
    """

    # filter list of trials that are in this batch
    batch_sim_data = data[batch_idx*batch_size:((batch_idx+1)*(batch_size))]

    # all sequences in the batch, attended item is coded as idx (as 0, 1, 2)
    if human_data:
        batch_fixation_sequences_idx = [(np.array(trial_data['fixations'])-1).tolist() for trial_data in batch_sim_data]
    else:
        batch_fixation_sequences_idx = [trial_data['fixations'] for trial_data in batch_sim_data]

    batch_choices_idx = [trial_data['choice'] - 1 for trial_data in batch_sim_data]

    # first 3 are fixation, last is choice...
    batch_sequences_cat = [[to_categorical(x, num_classes = 6) for x in this_sequence] for this_sequence in batch_fixation_sequences_idx]

    # now append to each of these the choice info - the choice gets it's own channel (of 3)
    batch_sequences_cat_w_choices = [batch_sequences_cat[i] + [to_categorical(3 + batch_choices_idx[i], num_classes = 6)] for i in range(len(batch_sequences_cat))]
    batch_data = pad_sequences(batch_sequences_cat_w_choices)
    batch_data = batch_data.astype('float32')

    batch_targets_init = np.array([trial_data['value'] for trial_data in batch_sim_data], dtype = 'float32')
    batch_targets_cont = [np.repeat([batch_targets_init[i]],len(batch_sequences_cat_w_choices[i]),axis=0) for i in range(len(batch_sequences_cat_w_choices))]

    batch_targets = pad_sequences(batch_targets_cont)
    batch_targets = batch_targets.astype('float32')
    
    return (batch_data, batch_targets)

# example_batch = gen_batch_data_fixations_choice(32, 0, human_data,human_data=True) # batch size = 32, idx = 0
# print(f'The return type is a {type(example_batch)} with length {len(example_batch)}.')
# print(f'The first item in the tuple is the batch of sequences with shape {example_batch[0].shape}.')
# print(f'The first element in the batch of sequences is:\n {example_batch[0][0, :, :]}')
# print(f'The second item in the tuple is the corresponding batch of targets with shape {example_batch[1].shape}.')
# print(f'The first element in the batch of targets is:\n {example_batch[1][0, :]}')


# this is for just choice
def gen_batch_data_choice_only(batch_size, batch_idx, data, human_data=False):
    # filter list of trials that are in this batch
    batch_sim_data = data[batch_idx*batch_size:((batch_idx+1)*(batch_size))]

    # all sequences in the batch, attended item is coded as idx (as 0, 1, 2)
    batch_choices_idx = [trial_data['choice'] - 1 for trial_data in batch_sim_data]

    batch_data = to_categorical(batch_choices_idx)
    batch_data = batch_data.astype('float32')

    batch_targets = np.array([trial_data['value'] for trial_data in batch_sim_data], dtype = 'float32')
    batch_targets= batch_targets.astype('float32')
    return (batch_data, batch_targets)

# example_batch = gen_batch_data_choice_only(32, 0, human_data,human_data=True) # batch size = 32, idx = 0
# print(f'The return type is a {type(example_batch)} with length {len(example_batch)}.')
# print(f'The first item in the tuple is the batch of choices with shape {example_batch[0].shape}.')
# print(f'The first element in the batch of sequences is:\n {example_batch[0][0, :]}')
# print(f'The second item in the tuple is the corresponding batch of targets with shape {example_batch[1].shape}.')
# print(f'The first element in the batch of targets is:\n {example_batch[1][0, :]}')


# this is for fixations only
def gen_batch_data_fixations_only(batch_size, batch_idx, data, human_data=False):

    """
    Create sequence and target data for a batch

    Input: 
        batch_size: number of trials to include in batch
        batch_idx: index of data
        data: list of dicts, where each dict has 'values', 'fixations', and 'choice'
        human_data: this is just coded differently, so need to specify

    Returns:
        a tuple, (batch_data, batch_targets)
        batch_data is 3d array: batch_size x sequence_size x one-hot categorical encoding (3 here)
        batch_targets is 2d array: 
    """

    # filter list of trials that are in this batch
    batch_sim_data = data[batch_idx*batch_size:((batch_idx+1)*(batch_size))]

    # all sequences in the batch, attended item is coded as idx (as 0, 1, 2)
    if human_data:
        batch_fixation_sequences_idx = [(np.array(trial_data['fixations'])-1).tolist() for trial_data in batch_sim_data]
    else:
        batch_fixation_sequences_idx = [trial_data['fixations'] for trial_data in batch_sim_data]

    # first 3 are fixation, last is choice...
    batch_sequences_cat = [[to_categorical(x, num_classes = 3) for x in this_sequence] for this_sequence in batch_fixation_sequences_idx]

    # now append to each of these the choice info - the choice gets it's own channel (of 3)
    batch_data = pad_sequences(batch_sequences_cat)
    batch_data = batch_data.astype('float32')

    batch_targets_init = np.array([trial_data['value'] for trial_data in batch_sim_data], dtype = 'float32')
    batch_targets_cont = [np.repeat([batch_targets_init[i]],len(batch_sequences_cat[i]),axis=0) for i in range(len(batch_sequences_cat))]

    batch_targets = pad_sequences(batch_targets_cont)
    batch_targets = batch_targets.astype('float32')
    
    return (batch_data, batch_targets)

# example_batch = gen_batch_data_fixations_only(32, 0, human_data,human_data=True) # batch size = 32, idx = 0
#print(f'The return type is a {type(example_batch)} with length {len(example_batch)}.')
# print(f'The first item in the tuple is the batch of sequences with shape {example_batch[0].shape}.')
# print(f'The first element in the batch of sequences is:\n {example_batch[0][0, :, :]}')
# print(f'The second item in the tuple is the corresponding batch of targets with shape {example_batch[1].shape}.')
# print(f'The first element in the batch of targets is:\n {example_batch[1][0, :]}')


# set up neural networks
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.lstm(x)[0]
        x = self.linear(h)
        return x
    
    def get_states_across_time(self, x):
        h_c = None
        h_list, c_list = list(), list()
        with torch.no_grad():
            for t in range(x.size(1)):
                h_c = self.lstm(x[:, [t], :], h_c)[1]
                h_list.append(h_c[0])
                c_list.append(h_c[1])
            h = torch.cat(h_list)
            c = torch.cat(c_list)
        return h, c
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_output = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        ha = self.input_hidden(x)
        hb = F.relu(ha)
        o = self.hidden_output(hb)
        return o


def test(model, test_sim_data, criterion, device, batch_size, n_total_seq, gen_batch_data):
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout.
    model.eval()
    
    n_total_seq = 50000

    n_batches = int(np.round(n_total_seq / batch_size));

    loss_res = np.zeros((n_batches, 1), dtype=float)

    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we typically don't need the gradients at this point.
    with torch.no_grad():
        for batch_idx in range(n_batches):
            data, target = gen_batch_data(batch_size, batch_idx, test_sim_data)
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            output = model(data)
            
            to_keep = target != 0
            target = target[to_keep]
            output = output[to_keep]
            
            
            # Pick only the output corresponding to last sequence element (input is pre padded)
            # output = output[:, -1, :]

            # target = target.argmax(dim=1)
            loss = criterion(output, target)  # is this just for the last batch?

            # store the loss
            loss_res[batch_idx] = loss.item()

    return np.mean(loss_res)

def train_and_test(model, train_sim_data, test_sim_data, criterion, optimizer, device, batch_size, n_total_seq, gen_batch_data, make_plot = False, model_name = ""):
    # Set the model to training mode. This will turn on layers that would
    # otherwise behave differently during evaluation, such as dropout.
    model.train()
    

    # how many batches
    n_batches = int(np.round(n_total_seq/batch_size));
    for batch_idx in range(n_batches):

        # Request a batch of sequences and class labels, convert them into tensors
        # of the correct type, and then send them to the appropriate device.
        #data, target = train_data_gen[batch_idx] # just alter this to the function that produces the data?
        data, target = gen_batch_data(batch_size, batch_idx, train_sim_data)
        
        # this needs to change... 
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

        # Perform the forward pass of the model
        output = model(data)  # Step ①

        
        # for some reason target is an int, and dosn't match the output which is float32
        target = target.to(torch.float32)
        
        # remove padding (nicely, this is just 0's)
        to_keep = target != 0
        target = target[to_keep]
        output = output[to_keep]
        
        # need to re-write this function... 
        loss = criterion(output, target)  # Step ②

        # Clear the gradient buffers of the optimized parameters.
        # Otherwise, gradients from the previous batch would be accumulated.
        optimizer.zero_grad()  # Step ③

        loss.backward()  # Step ④

        optimizer.step()  # Step ⑤
        
    # compute the test loss... 
    test_loss = test(model, test_data_sim, criterion, device, batch_size, n_total_seq, gen_batch_data)

    return model, test_loss#loss.item()

# Setup the training and test data generators
batch_size     = 32
n_total_seq = 1e6
n_batches = int(np.round(n_total_seq/batch_size));
n_tests = int(np.ceil(n_batches/250)) - 1

n_runs = 1
LSTM_run_losses = np.zeros((n_runs, n_tests))
for run_idx in range(n_runs):
    torch.manual_seed(run_idx)

    print(run_idx)

    # Setup the RNN and training settings
    input_size  = 6 # this is the length of the input vector? #train_data_gen.n_symbols
    hidden_size = 50
    output_size = 3 # this is the leågth of the output vector #train_data_gen.n_classes
    model       = SimpleLSTM(input_size, hidden_size, output_size)
    criterion   = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss()
    optimizer   = torch.optim.RMSprop(model.parameters(), lr=0.001)
    # optimizer   = torch.optim.Adam(model.parameters(), lr=0.00304)
    max_epochs  = 10
    device = torch.device('cpu')#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train the model›
    # model = train_and_test(model, train_data_gen, test_data_gen, criterion, optimizer, max_epochs)
    start_time = time.time()
    # model_LSTM = train_and_test(model, train_data_sim, test_data_sim, criterion, optimizer, max_epochs, batch_size, n_total_seq, verbose=True, model_name = 'LSTM')
    device = torch.device('cpu')#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_LSTM, loss_res, LSTM_batch_num = train_with_int_tests(model, train_data_sim, test_data_sim, criterion, optimizer, device, batch_size, n_total_seq, gen_batch_data, model_name='LSTM')
    LSTM_run_losses[run_idx,:] = loss_res


