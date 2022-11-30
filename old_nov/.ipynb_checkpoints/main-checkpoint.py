#####################
##### Packages ######
#####################

import numpy as np
import matplotlib.pyplot as plt
import pickle
import optuna
import time
import sys
import torch
import os
import argparse
import random

############################################
###### Load functions from other scripts 
############################################

from load_data_funs import load_data, gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only
from train_and_test_funs import test, train_on_simulation_then_human_with_intermediate_tests, test_record_each_output, compute_heldout_performance
from neural_nets import SimpleLSTM, SimpleMLP, SimpleGRU, SimpleTransformer

####################################################
#### Set cluster job_idx, folders and random seeds
#############################################

on_cluster = True

# set path to the human and simulation data
if on_cluster:
    sim_data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
    human_data_path = '/scratch/gpfs/erussek/RNN_project/human_trials.json'
else:
    sim_data_path = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'
    human_data_path = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/human_trials.json'

# set up folder to save results
if on_cluster:
    to_save_folder = '/scratch/gpfs/erussek/RNN_project/preference_model_results_nov20'
else:
    to_save_folder = '/Users/erussek/Dropbox/Griffiths_Lab_Stuff/Code/RNNs/preference_model_results_nov20'
    
if not os.path.exists(to_save_folder):
    os.mkdir(to_save_folder)

# initialize results dictionary
res_dict = {}

#############################################################################################
##### Read arguments and set options which depend on them (Want to add parameters to this) ##
#############################################################################################

parser = argparse.ArgumentParser(description='Run everything')

# which model to use
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of network (LSTM, GRU, Transformer, MLP)')

parser.add_argument('--train_seq_part', type=str, default = 'fix_and_choice',
                   help='use fixation data, choice data or both (fix_only,choice_only,fix_only')

# how many simulation sequences to train on?
parser.add_argument('--n_simulation_sequences_train', type=float, default = 1e3,
                   help='numƒber of simulation sequences to train on')

# how many epochs of human data (consisting of 2000 sequences) should we train on?
parser.add_argument('--n_human_epochs_train', type=float, default = 0,
                   help='how many epochs of human data to train on?')

# is test human or simualted?
parser.add_argument('--test_data_type', type=str, default = "simulation",
                   help='is test data simulation or human')

# how many sequences to test on (for either human or sim)?
parser.add_argument('--n_sequences_test', type=float, default = 1e3, # 1000
                   help='number of sequences to use for test')

# how many sequences to test on (for either human or sim)?
parser.add_argument('--n_sequences_final_performance', type=float, default = 1e3, # 1000
                   help='number of sequences for final (trained model) performance measures')


# model size / number of hidden units (for rnns / MLP) # vary this - 32 64 128 256
parser.add_argument('--d_model', type=float, default = 128, # 1000
                   help='model size / embedding size (Transformer) / number of hidden units (RNN/LSTM/MLP)')

# n_layers in Transformer , possibly add to LSTM (would this make sense though)?
parser.add_argument('--n_layers', type=float, default = 2, # 1000
                   help='Number of layers')

# n_head - only relevant for transformers
parser.add_argument('--n_head', type=float, default = 4, # 1000
                   help='number of sequences to use for test')

# learning rate
parser.add_argument('--lr', type=float, default = .001, # 1000
                   help='learning rate')

# learning rate
parser.add_argument('--batch_size', type=float, default = 32, # 1000
                   help='batch size')

# learning rate
parser.add_argument('--run_idx', type=int, default = 1, # 1000
                   help='batch size')


# Record key args
args = parser.parse_args()
print(args)
model_name = args.model
train_seq_part = args.train_seq_part
n_simulation_sequences_train = args.n_simulation_sequences_train
n_human_epochs_train = args.n_human_epochs_train
test_data_type = args.test_data_type
n_sequences_test = args.n_sequences_test
n_sequences_final_performance = args.n_sequences_final_performance
run_idx = args.run_idx


# model parameteres
d_model = args.d_model
n_layers = args.n_layers
n_head = args.n_head
lr = args.lr
batch_size = args.batch_size



# set the random seed.
random.seed(run_idx)
torch.manual_seed(run_idx)

# Functions to generate data
gen_data_func_by_train_seq_part = {"fix_and_choice": gen_batch_data_fixations_choice, "fix_only": gen_batch_data_fixations_only, "choice_only": gen_batch_data_choice_only}
gen_data_func = gen_data_func_by_train_seq_part[train_seq_part]

# Number of input tokens depends on task
n_tokens_by_train_seq_part = {"fix_and_choice": 6, "fix_only": 3, "choice_only": 3}
n_tokens = n_tokens_by_train_seq_part[train_seq_part]

######################
####### LOAD DATA ###
#####################
print("Loading Data")
train_data_sim, test_data_sim, train_data_human, test_data_human = load_data(sim_data_path, human_data_path,this_seed=run_idx,split_human_data=True)

##########################
#### SETUP MODEL #########
##########################

output_size = 3 # always return len 3 output size

if model_name == 'MLP': # this should only be called w/ choice only setting
    model       = SimpleMLP(n_tokens, d_model, output_size)
elif model_name == 'Transformer':    
    dim_feedforward = 4*d_model
    model = SimpleTransformer(n_tokens,d_model,dim_feedforward,output_size,nlayers = n_layers, nhead = n_head)
elif model_name == 'GRU':
    model = SimpleGRU(n_tokens, d_model, output_size)
elif model_name == 'LSTM':
    model = SimpleLSTM(n_tokens, d_model, output_size)
else:
    exit("Invalid model name entered")
    
    
print(model)

##################################
### OTHER TRAINING PARAMETERS ####
##################################
    
# non neural net training parameters
criterion   = torch.nn.MSELoss()
optimizer   = torch.optim.Adam(model.parameters(), lr=lr) # set learning rate...
start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#######################################################
##### Train the model and record learning curves #####
######################################################

print("Training the model")
simulation_loss_results, human_loss_results, train_sequence_number,human_sequence_number, simulation_sequence_number, model = train_on_simulation_then_human_with_intermediate_tests(model,train_data_sim, train_data_human,test_data_sim,test_data_human,criterion,optimizer,device,batch_size,n_simulation_sequences_train, n_human_epochs_train, n_sequences_test, gen_data_func)

# save results of this
res_dict["simulation_loss_results"] =  simulation_loss_results
res_dict["human_loss_results"] =  human_loss_results
res_dict["train_sequence_number"] =  train_sequence_number
res_dict["human_sequence_number"] =  human_sequence_number
res_dict["simulation_sequence_number"] =  simulation_sequence_number

############################################################################################
#### Evaluate model performance ##
############################################################################################

print("Evaluating trained model performance")
if train_seq_part != "choice_only":
    
    n_back_vals = np.arange(1,20)
    
    r_sim_by_n_back = np.zeros(len(n_back_vals))
    r_human_by_n_back = np.zeros(len(n_back_vals))
    
    pct_correct_max_sim_by_n_back = np.zeros(len(n_back_vals))
    pct_correct_max_human_by_n_back = np.zeros(len(n_back_vals))
    
    pct_correct_min_sim_by_n_back = np.zeros(len(n_back_vals))
    pct_correct_min_human_by_n_back = np.zeros(len(n_back_vals))
    
    pct_correct_order_sim_by_n_back = np.zeros(len(n_back_vals))
    pct_correct_order_human_by_n_back = np.zeros(len(n_back_vals))
    
    for nb_idx, nb in enumerate(n_back_vals):
        
        r_sim_by_n_back[nb_idx], pct_correct_max_sim_by_n_back[nb_idx], pct_correct_min_sim_by_n_back[nb_idx], pct_correct_order_sim_by_n_back[nb_idx] = compute_heldout_performance(model, test_data_sim, device, batch_size, n_sequences_final_performance,gen_data_func, nb, use_human_data=False)
        
        r_human_by_n_back[nb_idx], pct_correct_max_human_by_n_back[nb_idx], pct_correct_min_human_by_n_back[nb_idx], pct_correct_order_human_by_n_back[nb_idx] = compute_heldout_performance(model, test_data_human, device, batch_size, n_sequences_final_performance,gen_data_func, nb, use_human_data=True)
        
else:
    
    r_sim_by_n_back, pct_correct_max_sim_by_n_back, pct_correct_min_sim_by_n_back, pct_correct_order_sim_by_n_back = compute_heldout_performance(model, test_data_sim, device, batch_size, n_sequences_final_performance,gen_data_func, 0, choice_only=True, use_human_data=True)
    
    r_human_by_n_back, pct_correct_max_human_by_n_back, pct_correct_min_human_by_n_back, pct_correct_order_human_by_n_back = compute_heldout_performance(model, test_data_human, device, batch_size, n_sequences_final_performance, gen_data_func, 0, choice_only=True, use_human_data=True)
    

# store these results
res_dict["r_sim_by_n_back"] = r_sim_by_n_back
res_dict["pct_correct_max_sim_by_n_back"] = pct_correct_max_sim_by_n_back
res_dict["pct_correct_min_sim_by_n_back"] = pct_correct_min_sim_by_n_back
res_dict["pct_correct_order_sim_by_n_back"] = pct_correct_order_sim_by_n_back

res_dict["r_human_by_n_back"] = r_human_by_n_back
res_dict["pct_correct_max_human_by_n_back"] = pct_correct_max_human_by_n_back
res_dict["pct_correct_min_human_by_n_back"] = pct_correct_min_human_by_n_back
res_dict["pct_correct_order_human_by_n_back"] = pct_correct_order_human_by_n_back
    
########################
### SAVE RESULTS #######
########################

print("Saving the model and results")

# save the model with torch
model_file_name = 'model_model_name_{}_train_seq_part_{}_n_simulation_sequences_train_{}_n_human_epochs_train_{}_job_{}'.format(model_name,train_seq_part, n_simulation_sequences_train, n_human_epochs_train, run_idx)
model_full_file_name = os.path.join(to_save_folder, model_file_name)
torch.save(model, model_full_file_name)

# save the results dict with np.save 
res_file_name = 'res_model_name_{}_train_seq_part_{}_n_simulation_sequences_train_{}_n_human_epochs_train_{}_job_{}.pickle'.format(model_name,train_seq_part, n_simulation_sequences_train, n_human_epochs_train, run_idx)
loss_full_file_name = os.path.join(to_save_folder, res_file_name)

# save results file
with open(res_file_name, 'wb') as f:    
    pickle.dump(res_dict, f) 
