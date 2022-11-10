# set up packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys
import torch
import os
import random

from load_data_funs import load_data, gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only
from neural_nets import SimpleLSTM, SimpleMLP

train_setting = 0
job_idx = 0

on_cluster = False
if on_cluster:
    sim_data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
    human_data_path = '/scratch/gpfs/erussek/RNN_project/human_trials.json'
else:
    sim_data_path = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'
    human_data_path = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/human_trials.json'
    
train_data_funcs = [gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only]
this_data_func = train_data_funcs[train_setting]


# function to test model...
def test(model, test_sim_data, criterion, device, batch_size, n_total_seq, gen_batch_data,use_human_data = False):
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout.
    model.eval()
    
    n_total_seq = 1000

    n_batches = int(np.round(n_total_seq / batch_size));

    loss_res = np.zeros((n_batches, 1), dtype=float)

    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we typically don't need the gradients at this point.
    with torch.no_grad():
        for batch_idx in range(n_batches):
            data, target = gen_batch_data(batch_size, batch_idx, test_sim_data, use_human_data=use_human_data)
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            output = model(data)
            
            to_keep = target != 0
            target = target[to_keep]
            output = output[to_keep]
            
            # target = target.argmax(dim=1)
            loss = criterion(output, target)  # is this just for the last batch?

            # store the loss
            loss_res[batch_idx] = loss.item()

    return np.mean(loss_res)


def train_with_intermediate_tests(model, train_sim_data, test_sim_data, criterion, optimizer, device, batch_size, n_total_seq, gen_batch_data, use_human_data = False, model_name = "", n_epochs = 1):
    # Set the model to training mode. This will turn on layers that would
    # otherwise behave differently during evaluation, such as dropout.
    model.train()
    
    # What metric to store?
    # num_correct = 0

    # Iterate over every batch of sequences. Note that the length of a data generator
    # is defined as the number of batches required to produce a total of roughly 1000
    # sequences given a batch size.
        
    # how many batches
    n_batches = int(np.round(n_total_seq/batch_size));
    
    loss_res = []
    train_loss_res = []
    train_num = []
    
    print('n_epochs: '+str(n_epochs))
    
    for epoch_idx in range(n_epochs):
        print(epoch_idx)
        for batch_idx in range(n_batches):
            
            this_batch_idx = n_batches*epoch_idx + batch_idx
            #print(this_batch_idx)

            # Request a batch of sequences and class labels, convert them into tensors
            # of the correct type, and then send them to the appropriate device.
            data, target = gen_batch_data(batch_size, batch_idx, train_sim_data, use_human_data=use_human_data)

            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            # Perform the forward pass of the model
            output = model(data)  # Step


            # for some reason target is an int, and dosn't match the output which is float32
            target = target.to(torch.float32)

            # remove padding (nicely, this is just 0's)
            to_keep = target != 0
            target = target[to_keep]
            output = output[to_keep]

            # need to re-write this function... 
            loss = criterion(output, target)  # Step

            # Clear the gradient buffers of the optimized parameters.
            # Otherwise, gradients from the previous batch would be accumulated.
            optimizer.zero_grad()  # Step

            loss.backward()  # Step

            optimizer.step()  # Step

            # 
            if ((this_batch_idx % 100) == 0) & (batch_idx > 0):
                test_loss = test(model, test_sim_data, criterion, device, batch_size, n_total_seq, gen_batch_data, use_human_data=use_human_data)
                loss_res.append(test_loss)

                train_loss_res.append(loss.item())
                train_num.append(200*(this_batch_idx+1))
                
                print('batch num' + str(batch_idx) + ' loss: ' + str(test_loss))

        #return num_correct, loss.item()
    return model, np.array(loss_res), np.array(train_num)#loss.item()

train_data_sim, test_data_sim, human_data = load_data(sim_data_path, human_data_path,this_seed=job_idx)
this_data_func = train_data_funcs[train_setting]

# train on a 1.5 mil examples, generate learning curves... 
batch_size  = 32
n_total_seq = 1.5e5
n_batches = int(np.round(n_total_seq/batch_size));
n_tests = int(np.ceil(n_batches/200)) - 1
input_sizes = [6,3,3]
torch.manual_seed(job_idx)
input_size  = input_sizes[train_setting] # this is the length of the input vector? #train_data_gen.n_symbols
hidden_size = 50#best_hiddens[train_setting] # is this relevant for everything?
output_size = 3 # 

#### now create the model
import torch
import torch.nn as nn
import torch.nn.functional as F

# set up neural networks
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.gru(x)[0]
        x = self.linear(h)
        return x
    
# ?torch.nn.GRU
this_lr = .001

model = SimpleGRU(input_size,hidden_size,output_size)
criterion   = torch.nn.MSELoss()
optimizer   = torch.optim.RMSprop(model.parameters(), lr= this_lr) # switch to adam?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.train()

# What metric to store?
# num_correct = 0

# Iterate over every batch of sequences. Note that the length of a data generator
# is defined as the number of batches required to produce a total of roughly 1000
# sequences given a batch size.

# how many batches
n_batches = int(np.round(n_total_seq/batch_size));
n_epochs=1

loss_res = []
train_loss_res = []
train_num = []

print('n_epochs: '+str(n_epochs))
#epoch_idx = 1
#batch_idx = 1


#this_batch_idx = n_batches*epoch_idx + batch_idx
#print(this_batch_idx)

#gen_batch_data = this_data_func

# Request a batch of sequences and class labels, convert them into tensors
# of the correct type, and then send them to the appropriate device.
#data, target = gen_batch_data(batch_size, batch_idx, train_data_sim, use_human_data=False)
#data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)
#output = model(data)  # Step

trained_model, loss_res, train_num = train_with_intermediate_tests(model, train_data_sim, test_data_sim, criterion, optimizer, device, batch_size, n_total_seq, this_data_func, model_name='GRU')
