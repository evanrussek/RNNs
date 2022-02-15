import json
import numpy as np
import matplotlib.pyplot as plt
# funcs stolen from nyu deep learning course
from res.sequential_tasks import pad_sequences, to_categorical
import torch
import torch.nn as nn
import optuna
import time
import os
import pickle

on_cluster = True
n_optuna_trials = 350

# probably could get away with less memory?

# Functions to load data
def load_sim_data(data_path, n_train=1e5, n_test=1e5):
    """ Load simulated data from json file to dictionary """

    train_file_idxs = range(1,16)
    test_file_idxs = range(16,31)

    train_files = [os.path.join(data_path, str(i) + '.json') for i in train_file_idxs]
    test_files = [os.path.join(data_path, str(i) + '.json') for i in test_file_idxs]

    a = [json.load(open(train_files[i])) for i in range(15)]
    train_trials = [item for sublist in a for item in sublist]
    del a
    train_data_sim = train_trials[:int(n_train)]

    test_trials = json.load(open(train_files[0]))
    test_data_sim = test_trials[:int(n_test)]
    
    return train_data_sim, test_data_sim


def gen_batch_data(batch_size, batch_idx, sim_data):

    """
    Create sequence and target data for a batch

    Input: 
        batch_size: number of trials to include in batch
        batch_idx: index of data
        sim_data: list of dicts, where each dict has 'values', 'fixations', and 'choice'

    Returns:
        a tuple, (batch_data, batch_targets)
        batch_data is 3d array: batch_size x sequence_size x one-hot categorical encoding (3 here)
        batch_targets is 2d array: 
    """

    # filter list of trials that are in this batch
    batch_sim_data = sim_data[batch_idx*batch_size:((batch_idx+1)*(batch_size))]

    ## generate sequences of fixations + choice

    # all sequences in the batch, attended item is coded as idx (as 0, 1, 2)
    batch_sequences_idx = [trial_data['fixations'] + [trial_data['choice']-1] for trial_data in batch_sim_data]

    # all sequences in the batch, attended item coded as one-hot categorical: e.g. 0: [1,0,0] 1: [0,1,0], [0,0,1]
    batch_sequences_cat = [[to_categorical(x, num_classes = 3) for x in this_sequence] for this_sequence in batch_sequences_idx]

    # pad front of each sequence with n x [0,0,0] so that all seqeunces are same length
    batch_data = pad_sequences(batch_sequences_cat)
    batch_data = batch_data.astype('float32')


    ## generate sequences of targets
    batch_targets = np.array([trial_data['value'] for trial_data in batch_sim_data], dtype = 'float32')

    return (batch_data, batch_targets)


# Build neural net models

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # This just calls the base class constructor
        super().__init__()
        # Neural network layers assigned as attributes of a Module subclass
        # have their parameters registered for training automatically.
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        h = self.rnn(x)[0]
        x = self.linear(h)
        return x


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


# Define the Training loop -- edit this so we just go through once... 
def train(model, train_sim_data, criterion, optimizer, device, batch_size, n_total_seq):
    # Set the model to training mode. This will turn on layers that would
    # otherwise behave differently during evaluation, such as dropout.
    model.train()

    # What metric to store?
    # num_correct = 0

    # Iterate over every batch of sequences. Note that the length of a data generator
    # is defined as the number of batches required to produce a total of roughly 1000
    # sequences given a batch size.

    # how many batches
    n_batches = int(np.round(n_total_seq / batch_size))

    loss_res = np.zeros((n_batches, 1), dtype=float)

    for batch_idx in range(n_batches):
        # Request a batch of sequences and class labels, convert them into tensors
        # of the correct type, and then send them to the appropriate device.
        # data, target = train_data_gen[batch_idx] # just alter this to the function that produces the data?
        data, target = gen_batch_data(batch_size, batch_idx, train_sim_data)

        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

        # Perform the forward pass of the model
        output = model(data)  # Step ①

        # Pick only the output corresponding to last sequence element (input is pre padded)
        output = output[:, -1, :]

        # Compute the value of the loss for this batch. For loss functions like CrossEntropyLoss,
        # the second argument is actually expected to be a tensor of class indices rather than
        # one-hot encoded class labels. One approach is to take advantage of the one-hot encoding
        # of the target and call argmax along its second dimension to create a tensor of shape
        # (batch_size) containing the index of the class label that was hot for each sequence.

        # for some reason target is an int, and dosn't match the output which is float32
        target = target.to(torch.float32)
        loss = criterion(output, target)  # Step ②
        # store the loss
        loss_res[batch_idx] = loss.item()

        # Clear the gradient buffers of the optimized parameters.
        # Otherwise, gradients from the previous batch would be accumulated.
        optimizer.zero_grad()  # Step ③

        loss.backward()  # Step ④

        optimizer.step()  # Step ⑤

        # y_pred = output.argmax(dim=1)

        # this is wrong since we're doing regression...
        # num_correct += (y_pred == target).sum().item()

    # return num_correct, loss.item()
    return np.mean(loss_res)  # loss.item()


# Define the testing loop
def test(model, test_sim_data, criterion, device, batch_size, n_total_seq):
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout.
    model.eval()

    # Store the number of sequences that were classified correctly
    # num_correct = 0

    n_batches = int(np.round(n_total_seq / batch_size));

    loss_res = np.zeros((n_batches, 1), dtype=float)

    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we typically don't need the gradients at this point.
    with torch.no_grad():
        for batch_idx in range(n_batches):
            data, target = gen_batch_data(batch_size, batch_idx, test_sim_data)
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            output = model(data)
            # Pick only the output corresponding to last sequence element (input is pre padded)
            output = output[:, -1, :]

            # target = target.argmax(dim=1)
            loss = criterion(output, target)  # is this just for the last batch?

            # store the loss
            loss_res[batch_idx] = loss.item()

            # y_pred = output.argmax(dim=1)
            # num_correct += (y_pred == target).sum().item()

    # return num_correct, loss.item()
    # print(loss_res)
    # print(loss.item())

    return np.mean(loss_res)  # loss.item()


# Train and test
def train_and_test(model_in, train_data_sim_in, test_data_sim_in, criterion_in, optimizer, max_epochs, batch_size,
                   n_total_seq_in,
                   verbose=True, model_name='', make_plot=False):
    # Automatically determine the device that PyTorch should use for computation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Move model to the device which will be used for train and test
    model_in.to(device)

    # Track the value of the loss function and model accuracy across epochs
    # history_train = {'loss': [], 'acc': []}
    # history_test = {'loss': [], 'acc': []}

    history_train = {'loss': []}
    history_test = {'loss': []}

    for epoch in range(max_epochs):
        # Run the training loop and calculate the accuracy.
        # Remember that the length of a data generator is the number of batches,
        # so we multiply it by the batch size to recover the total number of sequences.
        # num_correct, loss = train(model, train_data_gen, criterion, optimizer, device)
        loss = train(model_in, train_data_sim_in, criterion_in, optimizer, device, batch_size, n_total_seq_in)
        # accuracy = float(num_correct) / (len(train_data_gen) * train_data_gen.batch_size) * 100
        history_train['loss'].append(loss)
        # history_train['acc'].append(accuracy)

        # Do the same for the testing loop
        # num_correct, loss = test(model, test_data_gen, criterion, device)
        loss = test(model_in, test_data_sim_in, criterion_in, device, batch_size, n_total_seq_in)
        history_test['loss'].append(loss)

        if (verbose & (epoch % 10 == 0)) or epoch + 1 == max_epochs:
            print(f'[Epoch {epoch + 1}/{max_epochs}]'
                  f" loss: {history_train['loss'][-1]:.4f} (MSE)"
                  f" - test_loss: {history_test['loss'][-1]:.4f} (MSE)")

    if make_plot:
        # Generate diagnostic plots for the loss and accuracy
        fig, ax = plt.subplots(1, figsize=(4.4, 4.5))
        # for ax, metric in zip(axes, ['loss', 'acc']):
        ax.plot(history_train['loss'])
        ax.plot(history_test['loss'])
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel('loss (MSE)', fontsize=12)
        ax.legend(['Train', 'Test'], loc='best')
        ax.set_title(model_name)
        plt.show(block=False)

    return model_in


# evaluate model


# if __name__ == '__main__':

# Set the random seed for reproducible results
torch.manual_seed(1)


def objective(trial):
    
    if on_cluster:
        data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
    else:
        data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'

    (train_data_sim, test_data_sim) = load_sim_data(data_path, n_train=1e5, n_test=1e5)

    # Set up the training and test data generators
    batch_size = 32  # trial.suggest_int('batch_size', 10, 100)  # 32
    n_total_seq = 1e5

    # Set up the RNN and training settings
    input_size = 3  # this is the length of the input vector? #train_data_gen.n_symbols
    hidden_size = trial.suggest_int('hidden_size', 2, 200, step=5)  # 4
    output_size = 3  # this is the leågth of the output vector #train_data_gen.n_classes
    # model = SimpleRNN(input_size, hidden_size, output_size)
    model = SimpleLSTM(input_size, hidden_size, output_size)

    criterion = torch.nn.MSELoss()  # torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=trial.suggest_float('lr', .0001, .01, log=True))  # was previously .001
    max_epochs = 1  # just 1 epoch... 

    # Train the model
    model_rnn = train_and_test(model, train_data_sim, test_data_sim, criterion, optimizer, max_epochs, batch_size,
                               n_total_seq, verbose=False, model_name='LSTM', make_plot=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = test(model_rnn, test_data_sim, criterion, device, batch_size, n_total_seq)

    return loss


def run_with_default_settings(on_cluster=False):
    
    # add change here for if we're on the cluster
    if on_cluster:
        data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
    else:
        data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'


    (train_data_sim, test_data_sim) = load_sim_data(data_path, n_train=1e5, n_test=1e5)

    # Set up the training and test data generators
    batch_size = 32  # trial.suggest_int('batch_size', 10, 100)  # 32
    n_total_seq = 100

    # Set up the RNN and training settings
    input_size = 3  # this is the length of the input vector? #train_data_gen.n_symbols
    hidden_size = 4
    output_size = 3  # this is the leågth of the output vector #train_data_gen.n_classes
    # model = SimpleRNN(input_size, hidden_size, output_size)
    model = SimpleLSTM(input_size, hidden_size, output_size)

    criterion = torch.nn.MSELoss()  # torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=.001)
    max_epochs = 1  # let's just set it to this to start

    # Train the model
    model_rnn = train_and_test(model, train_data_sim, test_data_sim, criterion, optimizer, max_epochs, batch_size,
                               n_total_seq, verbose=False, model_name='LSTM', make_plot=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = test(model_rnn, test_data_sim, criterion, device, batch_size, n_total_seq)

    
def save_study(study, file_name, on_cluster = False):

    if on_cluster:
        to_save_folder = '/scratch/gpfs/erussek/RNN_project/optuna_results'
    else:
        to_save_folder = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Code/RNNs/optuna_results'

    if not os.path.exists(to_save_folder):
        os.mkdir(to_save_folder)
    
    to_save_file = os.path.join(to_save_folder, file_name)
    
    outfile = open(to_save_file,'wb')
    pickle.dump(study,outfile)
    outfile.close()
    

if __name__ == '__main__':
    study = optuna.create_study()
    start_time = time.time()
    study.optimize(objective, n_trials=n_optuna_trials)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    save_study(study, 'optimal_sims1_optuna_results', on_cluster = on_cluster)
