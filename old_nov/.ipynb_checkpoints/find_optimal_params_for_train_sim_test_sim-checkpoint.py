# set up packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import optuna
import time
import sys
import torch
import os
import argparse


# add possibility for transformer to separate encoding dim from input dim 
# - this might be necessary 

# think this is working.. can we add an option now for using human data?

from load_data_funs import load_data, gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only
from train_and_test_funs import test, train_and_test
from neural_nets import SimpleLSTM, SimpleMLP, SimpleGRU, SimpleTransformer

# Read in arguments for script
parser = argparse.ArgumentParser(description='Find optimal parameters for neural network train and test on sim')

parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (LSTM, GRU, Transformer, MLP)')

parser.add_argument('--train_seq_part', type=str, default = 'fix_and_choice',
                   help='use fixation data, choice data or both (fix_only,choice_only,fix_only')

# train on sim, human or mixture
parser.add_argument('--train_data_type', type=str, default = 'simulated',
                   help='train on simulated data (simulated), human data (human), or mixture (mixture)')

parser.add_argument('--test_data_type', type=str, default = 'simulated',
                   help='train on simulated data (simulated), human data (human), or mixture (mixture)')

parser.add_argument('--n_optuna_trials', type=int, default = 150,
                   help='number of optuna trials to run')

parser.add_argument('--n_runs', type=int, default = 4,
                   help='number of runs to average for each optuna trials')

parser.add_argument('--n_seq_train', type=float, default = 1e4,
                   help='number of sequences to train')

parser.add_argument('--n_seq_test', type=float, default = 3e3,
                   help='number of sequences to use for test')

args = parser.parse_args()
this_train_setting_name = args.train_data_seq_part # use choice 
this_model_name = args.model
n_runs = args.n_runs
n_seq_train = args.n_seq_train
n_seq_test = args.n_seq_test

print('Model: {}, Train Data: {}'.format(this_model_name, this_train_setting_name))

# set up data generation functions...
train_data_funcs = {"fix_and_choice": gen_batch_data_fixations_choice, "fix_only": gen_batch_data_fixations_only, "choice_only": gen_batch_data_choice_only}
this_data_func = train_data_funcs[this_train_setting_name]
input_sizes = {"fix_and_choice": 6, "fix_only": 3, "choice_only": 3} # input size depends on 

# Setup the training and test data generators

# path to data
on_cluster = True
if on_cluster:
    sim_data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
    human_data_path = '/scratch/gpfs/erussek/RNN_project/human_trials.json'
else:
    sim_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'
    human_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/human_trials.json'

def objective(trial, train_data_sim, test_data_sim, this_train_setting_name):
        
    batch_size   = 32
    
    run_losses = np.zeros((n_runs))
    for run_idx in range(n_runs):
        
        # set a seed
        torch.manual_seed(run_idx)

        # Setup the RNN and training settings
        input_size  = input_sizes[this_train_setting_name] # this is the length of the input vector? #train_data_gen.n_symbols
        hidden_size = trial.suggest_int('hidden_size', 2, 200, step=5)  
        output_size = 3
        
        if this_model_name == 'MLP': # this should only be called w/ choice only setting
            model       = SimpleMLP(input_size, hidden_size, output_size)
        elif this_model_name == 'Transformer':
            
            # should these be tested out as well?
            nlayers= 1
            nhead = 1
            dropout=.1
                        
            model = SimpleTransformer(input_size,hidden_size,output_size,nlayers = 1, nhead = 1, dropout=.1)
        elif this_model_name == 'GRU':
            model = SimpleGRU(input_size, hidden_size, output_size)
        elif this_model_name == 'LSTM':
            model = SimpleLSTM(input_size, hidden_size, output_size)
        else:
            exit("Invalid model name entered")
        
        criterion   = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss()
        optimizer   = torch.optim.Adam(model.parameters(), lr=trial.suggest_float('lr', .0001, .01, log=True)) # was previously .001
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data_func = this_data_func

        # Train the model
        start_time = time.time()
        model_LSTM, loss = train_and_test(model, train_data_sim, test_data_sim, criterion, optimizer, device, batch_size, n_seq_train, n_seq_test, data_func, model_name=this_model_name)
        run_losses[run_idx]=loss
    return np.mean(run_losses)

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
    
    
    # load data... 
    sim_train_data, sim_test_data, human_train_data, human_test_data = load_data(sim_data_path, human_data_path, split_human_data = true)
    
    study = optuna.create_study()
    start_time = time.time()
    
    this_obj = lambda trial: objective(trial, train_data, test_data, this_train_setting_name)
    
    study.optimize(this_obj, n_trials=args.n_optuna_trials)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    this_file_name = 'optima_res_{}_{}_.pkl'.format(this_model_name, this_train_setting_name)
    
    # set the name for this study.. better!!!
    save_study(study, this_file_name, on_cluster = on_cluster)
    

