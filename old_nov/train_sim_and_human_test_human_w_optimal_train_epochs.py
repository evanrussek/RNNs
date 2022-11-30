# set up packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import optuna
import time
import sys
import torch
import os
import random

from load_data_funs import load_data, gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only
from neural_nets import SimpleLSTM, SimpleMLP

# get job from cluster (we can run 50)... 

is_array_job=True
on_cluster=True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1 # this is 0 to 50
    train_setting= int(sys.argv[1])
else:
    job_idx = 0
    train_setting=2
    prop_pretrain_setting = 1 # some number 1-5

# set the random seed.
random.seed(job_idx)

if on_cluster:
    sim_data_path = '/scratch/gpfs/erussek/RNN_project/optimal_fixation_sims'
    human_data_path = '/scratch/gpfs/erussek/RNN_project/human_trials.json'
else:
    sim_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/optimal_fixation_sims'
    human_data_path = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Data/RNNs/human_trials.json'
    

train_data_funcs = [gen_batch_data_fixations_choice, gen_batch_data_fixations_only, gen_batch_data_choice_only]
this_data_func = train_data_funcs[train_setting]

#get best learning rates from optuna search - for choice only it didn't matter


# vary how much pre-training data to use


best_lrs = [0.0019260129757659558, 0.0044066090959512735, .001]# 0.0001002995005652193]
best_hiddens = [97, 37, 50]

#best_n_pretrain = [7.5e5, 7.5e5, 0] # with pretraiing
best_n_pretrain = [0, 0, 0] # try with no pretraining

# best_n_train = [8432, 5232, 500000] # with pretraining
best_n_train = [64032, 32032, 924832] #  with no pretraining

# function to test model...
def test(model, test_sim_data, criterion, device, batch_size, gen_batch_data,human_data = False):
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
            data, target = gen_batch_data(batch_size, batch_idx, test_sim_data, human_data=human_data)
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


def train_sim_then_human_with_intermediate_tests(model, train_data_sim, train_data_human, test_data_sim, test_data_human, criterion, optimizer, device, batch_size, n_sim_seq, n_human_epochs, gen_batch_data):

    n_human_seq = len(human_train_data) # about 2000

    n_batches_sim = int(np.round(n_sim_seq/batch_size));
    n_batches_human = int(np.round(n_human_seq/batch_size));
    # first train on sim data...

    sim_loss_res=[]
    human_loss_res=[]
    train_num=[]


    # train on simulated data...
    print('Training on simulated data')
    for batch_idx in range(n_batches_sim):
        
        # put model in train mode (test should set it to test mode?)
        model.train()

        # Request a batch of sequences and class labels, convert them into tensors
        # of the correct type, and then send them to the appropriate device.
        data, target = gen_batch_data(batch_size, batch_idx, train_data_sim, human_data=False)
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

        # Perform the forward pass of the model
        output = model(data)

        # make sure target is correct type
        target = target.to(torch.float32)

        # filter out padding
        to_keep = target != 0
        target = target[to_keep]
        output = output[to_keep]

        # compute loss and backpropogate
        loss = criterion(output, target)  # Step
        optimizer.zero_grad()  # Step
        loss.backward()  # Step
        optimizer.step()  # Step

        # test every 100 batches... 

        # compute loss on both sim and human

        if ((batch_idx % 100) == 0) & (batch_idx > 0):
            sim_test_loss = test(model, test_data_sim, criterion, device, batch_size, gen_batch_data, human_data=False)
            sim_loss_res.append(sim_test_loss)

            human_test_loss = test(model, test_data_human, criterion, device, batch_size, gen_batch_data, human_data=True)
            human_loss_res.append(human_test_loss)

            train_num.append(32*(batch_idx+1))

            print('batch num' + str(batch_idx) + ' sim test loss: ' + str(sim_test_loss) + ' human test loss ' + str(human_test_loss))


    # now train on human data
    print('Training on human data')
    for epoch_idx in range(n_human_epochs):
        print('Human epoch: {}'.format(epoch_idx))
        for batch_idx in range(n_batches_human):

            this_batch_idx = n_batches_sim + n_batches_human*epoch_idx + batch_idx

            # Request a batch of sequences and class labels, convert them into tensors
            # of the correct type, and then send them to the appropriate device.
            data, target = gen_batch_data(batch_size, batch_idx, train_data_human, human_data=True)
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            # Perform the forward pass of the model
            output = model(data)

            # make sure target is correct type
            target = target.to(torch.float32)

            # filter out padding
            to_keep = target != 0
            target = target[to_keep]
            output = output[to_keep]

            # compute loss and backpropogate
            loss = criterion(output, target)  # Step
            optimizer.zero_grad()  # Step
            loss.backward()  # Step
            optimizer.step()  # Step

            # test every 100 batches... 

            # compute loss on both sim and human

            if ((this_batch_idx % 100) == 0) & (batch_idx > 0):
                sim_test_loss = test(model, test_data_sim, criterion, device, batch_size, gen_batch_data, human_data=False)
                sim_loss_res.append(sim_test_loss)

                human_test_loss = test(model, test_data_human, criterion, device, batch_size, gen_batch_data, human_data=True)
                human_loss_res.append(human_test_loss)

                train_num.append(32*(this_batch_idx+1))

                print('batch num' + str(batch_idx) + ' sim test loss: ' + str(sim_test_loss) + ' human test loss ' + str(human_test_loss))
            
    return np.array(sim_loss_res), np.array(human_loss_res), np.array(train_num), model

# compoute correlation with held-out test data... 
def test_record_each_output(model, test_sim_data, device, batch_size, n_total_seq, gen_batch_data,out_idx, choice_only=False, human_data=False):
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout.
    
    # print(choice_only)
    model.eval()

    # Store the number of sequences that were classified correctly
    # num_correct = 0

    n_batches = int(np.round(n_total_seq/batch_size));

    output_all = np.zeros((0,3))
    target_all = np.zeros((0,3))

    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we typically don't need the gradients at this point.
    with torch.no_grad():
        for batch_idx in range(n_batches):
            data, target = gen_batch_data(batch_size, batch_idx, test_sim_data, human_data = human_data)
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            output = model(data)
            # Pick only the output corresponding to last sequence element (input is pre padded)
            if not choice_only:
                output = output[:, -out_idx, :]
                target = target[:,-out_idx,:]

            output_all = np.concatenate((output_all, output.numpy()))
            target_all = np.concatenate((target_all, target.numpy()))

    return (output_all, target_all)


def compute_heldout_correlation(trained_model, test_data_sim, device, batch_size, n_seq_test,this_data_func, n_back, choice_only=False, human_data=False):
    
    trained_model.eval()
    
    output_all, target_all = test_record_each_output(trained_model, test_data_sim, device, batch_size, n_seq_test,this_data_func, n_back,choice_only=choice_only, human_data=human_data)
    output_flat = output_all.flatten()
    target_flat = target_all.flatten()
    output_flat = output_flat[target_flat != 0]
    target_flat = target_flat[target_flat != 0]
    return np.corrcoef(output_flat, target_flat)[1][0]


if __name__ == '__main__':
    
    
    # set up folder to save results
    if on_cluster:
        to_save_folder = '/scratch/gpfs/erussek/RNN_project/train_on_sim_and_human_results_optimal_no_pt'
    else:
        to_save_folder = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Code/RNNs/train_on_sim_and_human_results_optimal_no_pt'

    if not os.path.exists(to_save_folder):
        os.mkdir(to_save_folder)

    # load data
    train_data_sim, test_data_sim, human_train_data,human_test_data = load_data(sim_data_path, human_data_path,this_seed=job_idx,split_human_data=True)
    this_data_func = train_data_funcs[train_setting]
    
    # want to pretrain on some number of (1 epoch only) sequences of train_sim_data, and then train fully on some number of epochs of human_test_data...
    # train on a 1 mil. examples, generate learning curves... 
    batch_size  = 32
#    n_batches = int(np.round(n_total_seq/batch_size));
#    n_tests = int(np.ceil(n_batches/200)) - 1

    input_sizes = [6,3,3]

    torch.manual_seed(job_idx)

    input_size  = input_sizes[train_setting] # this is the length of the input vector? #train_data_gen.n_symbols
    hidden_size = best_hiddens[train_setting]
    output_size = 3 # 

    if train_setting == 2:
        model       = SimpleMLP(input_size, hidden_size, output_size)
    else:
        model       = SimpleLSTM(input_size, hidden_size, output_size)

    criterion   = torch.nn.MSELoss()
    optimizer   = torch.optim.RMSprop(model.parameters(), lr=best_lrs[train_setting])
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
 #   best_n_pretrain = [7.5e5, 7.5e5, 0]
#best_n_train = [8432, 5232, 500000]
    
    n_sim_seq = best_n_pretrain[train_setting]; # split this between 
    n_human_epochs = int(np.ceil(best_n_train[train_setting]/len(human_train_data))) # each epoch is approx a pass through the data
    gen_batch_data = this_data_func
    train_data_sim = train_data_sim
    train_data_human = human_train_data
    test_data_sim = test_data_sim
    test_data_human = human_test_data

    sim_loss_res, human_loss_res, train_num, trained_model = train_sim_then_human_with_intermediate_tests(model, train_data_sim, train_data_human, test_data_sim, test_data_human, criterion, optimizer, device, batch_size, n_sim_seq, n_human_epochs, gen_batch_data)
    
    # save loss curve
    loss_file_name = 'loss_res_train_setting_{}_job_{}.npy'.format(train_setting,job_idx)
    loss_full_file_name = os.path.join(to_save_folder, loss_file_name)

    # save model
    model_full_file_name = os.path.join(to_save_folder, 'model_train_setting_{}_job_{}'.format(train_setting,job_idx))
    torch.save(trained_model, model_full_file_name)
    
    # compute predictive accuracy on held-out simulated data (r)
    n_seq_test = 1e3
    if train_setting < 2:
        n_back_vals = np.arange(1,20)
        r_sim_by_n_back = np.zeros(len(n_back_vals))
        r_human_by_n_back = np.zeros(len(n_back_vals))
        for nb_idx, nb in enumerate(n_back_vals):
            r_sim_by_n_back[nb_idx] = compute_heldout_correlation(trained_model, test_data_sim, device, batch_size, n_seq_test,this_data_func, nb)
            r_human_by_n_back[nb_idx] = compute_heldout_correlation(trained_model, human_test_data, device, batch_size, n_seq_test,this_data_func, nb, human_data=True)
    else:
        r_sim_by_n_back = compute_heldout_correlation(trained_model, test_data_sim, device, batch_size, n_seq_test,this_data_func, 0, choice_only=True)
        #human_data_func = lambda x, y, z: this_data_func(x,y,z, human_data=True)
        r_human_by_n_back = compute_heldout_correlation(trained_model, human_test_data, device, batch_size, n_seq_test,this_data_func, 0, choice_only=True, human_data=True)

    # now compute the predictive accuracy on human data (r)...

    # save r and mse... 
    with open(loss_full_file_name, 'wb') as f:
        np.save(f, sim_loss_res)
        np.save(f, human_loss_res)
        np.save(f, train_num)
        np.save(f, r_sim_by_n_back)
        np.save(f, r_human_by_n_back)
