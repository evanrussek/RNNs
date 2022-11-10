# set up packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import optuna
import time
import sys
import torch


from load_data_funs import load_data, gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only
from neural_nets import SimpleLSTM, SimpleMLP, simpleGRU

on_cluster = False
n_optuna_trials = 150 # is this enough?

train_setting= 0 #int(sys.argv[1])
RNN_setting = 1 # 0 means LSTM, 1 means GRU

if on_cluster:
    sim_data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
    human_data_path = '/scratch/gpfs/erussek/RNN_project/human_trials.json'
else:
    sim_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'
    human_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/human_trials.json'

train_setting_names = ["fix_and_choice", "fix_only", "choice_only"]
this_setting_name = train_setting_names[train_setting]
RNN_models = [SimpleLSTM, SimpleGRU]
thisRNNModel = RNN_models[]


train_data_funcs = [gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only]
this_data_func = train_data_funcs[train_setting]

def test(model, test_sim_data, criterion, device, batch_size, n_total_seq, gen_batch_data):
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout.
    model.to(device)

    
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
    
    model.to(device)

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
        
    # compute the test loss... 
    test_loss = test(model, test_data_sim, criterion, device, batch_size, n_total_seq, gen_batch_data)

    return model, test_loss#loss.item()

# Setup the training and test data generators
# Setup the training and test data generators

def objective(trial, train_data_sim, test_data_sim, train_setting):
    
    input_sizes = [6,3,3]
    
    batch_size   = 32
    n_total_seq = 1e5 # could you do it for longer?

    n_runs = 4
    run_losses = np.zeros((n_runs))
    for run_idx in range(n_runs):
        torch.manual_seed(run_idx)

        print(run_idx)

        # Setup the RNN and training settings
        input_size  = input_sizes[train_setting] # this is the length of the input vector? #train_data_gen.n_symbols
        hidden_size = trial.suggest_int('hidden_size', 2, 200, step=5)  
        output_size = 3 # 
        if train_setting == 2:
            model       = SimpleMLP(input_size, hidden_size, output_size)
        else:
            #if RNN_setting == 0:
                model       = thisRNNModel(input_size, hidden_size, output_size)
            #else
        
        criterion   = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss()
        optimizer   = torch.optim.RMSprop(model.parameters(), lr=trial.suggest_float('lr', .0001, .01, log=True)) # was previously .001
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data_func = this_data_func

        # Train the model
        start_time = time.time()
        model_LSTM, loss = train_and_test(model, train_data_sim, test_data_sim, criterion, optimizer, device, batch_size, n_total_seq, data_func, model_name='LSTM')
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

# ... 
if __name__ == '__main__':
    
    
    train_data_sim, test_data_sim, human_data = load_data(sim_data_path, human_data_path)
    
    study = optuna.create_study()
    start_time = time.time()
    
    this_obj = lambda trial: objective(trial, train_data_sim, test_data_sim, train_setting)
    
    study.optimize(this_obj, n_trials=n_optuna_trials)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    save_study(study, 'optimal_'+this_setting_name+'.pkl', on_cluster = on_cluster)
    

