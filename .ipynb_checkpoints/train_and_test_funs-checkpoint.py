import numpy as np
import torch

# is the same function used for this accross scripts?
def test(model, test_data, criterion, device, batch_size, n_total_seq, gen_batch_data, use_human_data = False):
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout.
    model.to(device)

    model.eval()
    
    n_batches = int(np.round(n_total_seq / batch_size));

    loss_res = np.zeros((n_batches, 1), dtype=float)

    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we typically don't need the gradients at this point.
    with torch.no_grad():
        for batch_idx in range(n_batches):
            data, target = gen_batch_data(batch_size, batch_idx, test_data, use_human_data=use_human_data)
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


# add option to train on both human and sim?
def train_and_test(model, train_data_sim, test_data_sim,criterion, optimizer, device, batch_size, n_total_seq_train, n_total_seq_test, gen_batch_data, make_plot = False, model_name = ""):
    # Set the model to training mode. This will turn on layers that would
    # otherwise behave differently during evaluation, such as dropout.
    
    # send model to device if it hasn't yet
    model.to(device)

    # set model to train mode
    model.train()
    
    # how many batches will we run?
    n_batches = int(np.round(n_total_seq_train/batch_size));
    for batch_idx in range(n_batches):

        # Request a batch of sequences and class labels, convert them into tensors
        # of the correct type, and then send them to the appropriate device.
        data, target = gen_batch_data(batch_size, batch_idx, train_data)        
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
    test_loss = test(model, test_data, criterion, device, batch_size, n_total_seq_test, gen_batch_data)

    return model, test_loss#loss.item()



def train_on_simulation_then_human_with_intermediate_tests(model, train_data_sim, train_data_human, test_data_sim, test_data_human, criterion, device, batch_size, n_simulation_sequences_train, n_human_sequences_train, n_sequences_test, gen_batch_data, sim_lr = .001, human_lr = .001, test_batch_increment_sim = 100, test_batch_increment_human = 100):

    # move the model to the device
    model.to(device)
    
    # number of batches to run to get through all simulation sequences
    n_batches_simulation = int(np.round(n_simulation_sequences_train/batch_size));
    
    # number of batches to get through all human sequences
    n_human_sequences = len(train_data_human)
    n_batches_human = int(np.round(n_human_sequences/batch_size));

    # initialize lists to store results
    simulation_loss_results =[]
    human_loss_results= []
    train_sequence_number = []
    simulation_sequence_number = []
    human_sequence_number = []

    # Train on simulated data
    print('Training on simulated data')
    
    num_simulation_sequences_so_far = 0
    num_human_sequences_so_far = 0
    
    optimizer   = torch.optim.Adam(model.parameters(), lr=sim_lr) # set learning rate...
    
    
    for batch_idx in range(n_batches_simulation):
                
        # set model to train mode
        model.train()
        
        # update model, running a single batch on simulation data
        model = run_batch_update_model(model,train_data_sim, gen_batch_data, batch_size,batch_idx, optimizer, criterion, device, use_human_data=False)

        # if we're on a test increment, test the model and record the results
        if ((batch_idx % test_batch_increment_sim) == 0) & (batch_idx > 0):
            
            # get loss on simulation data
            sim_test_loss = test(model, test_data_sim, criterion, device, batch_size, n_sequences_test ,gen_batch_data, use_human_data=False)
            simulation_loss_results.append(sim_test_loss)

            # get loss on human data
            human_test_loss = test(model, test_data_human, criterion, device, batch_size,n_sequences_test, gen_batch_data, use_human_data=True)
            human_loss_results.append(human_test_loss)

            # record number of sequences traind on so far
            num_simulation_sequences_so_far = batch_size*(batch_idx+1)
            
            simulation_sequence_number.append(num_simulation_sequences_so_far)
            human_sequence_number.append(num_human_sequences_so_far)
            train_sequence_number.append(num_simulation_sequences_so_far)

            print('number of simulation seqeuences: ' + str(num_simulation_sequences_so_far) + 'number of human seqeuences: ' + str(num_human_sequences_so_far) + ' sim test loss: ' + str(sim_test_loss) + ' human test loss ' + str(human_test_loss))

    # now train on human data
    print('Training on human data')
    
    
    optimizer   = torch.optim.Adam(model.parameters(), lr=human_lr) # set learning rate...

    
    n_human_epochs_train = int(np.ceil(n_human_sequences_train/len(train_data_human))) # each epoch is approx a pass through the data
    for epoch_idx in range(n_human_epochs_train):
        #print('Human epoch: {}'.format(epoch_idx))
        for batch_idx in range(n_batches_human):
            # print(batch_idx)

            # set model to train mode
            model.train()
            
            # update model, running a single batch on human data -- lol found the bug!
            model = run_batch_update_model(model,train_data_human, gen_batch_data, batch_size,batch_idx, optimizer, criterion, device, use_human_data=True)

            current_batch_idx_human = n_batches_human*epoch_idx + batch_idx
            if ((current_batch_idx_human % test_batch_increment_human) == 0) & (batch_idx > 0):
                
                # get loss on simulation data
                sim_test_loss = test(model, test_data_sim, criterion, device, batch_size, n_sequences_test ,gen_batch_data, use_human_data=False)
                simulation_loss_results.append(sim_test_loss)

                # get loss on human data
                human_test_loss = test(model, test_data_human, criterion, device, batch_size,n_sequences_test, gen_batch_data, use_human_data=True)
                human_loss_results.append(human_test_loss)
                
                num_human_sequences_so_far = batch_size*(n_batches_human*epoch_idx + batch_idx)
                simulation_sequence_number.append(num_simulation_sequences_so_far)
                human_sequence_number.append(num_human_sequences_so_far)
                train_sequence_number.append(num_simulation_sequences_so_far + num_human_sequences_so_far)

                print('number of simulation seqeuences: ' + str(num_simulation_sequences_so_far) + 'number of human seqeuences: ' + str(num_human_sequences_so_far) + ' sim test loss: ' + str(sim_test_loss) + ' human test loss ' + str(human_test_loss))
    
    return np.array(simulation_loss_results), np.array(human_loss_results), np.array(train_sequence_number), np.array(human_sequence_number), np.array(simulation_sequence_number), model


def run_batch_update_model(model,train_data, gen_batch_data, batch_size,batch_idx, optimizer, criterion, device, use_human_data=False):

    # Request a batch of sequences and class labels, convert them into tensors
    # of the correct type, and then send them to the appropriate device.
    data, target = gen_batch_data(batch_size, batch_idx, train_data, use_human_data=use_human_data)
    data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

    # Perform the forward pass of the model
    output = model(data) # step 1

    # make sure target is correct type
    target = target.to(torch.float32)

    # filter out padding
    to_keep = target != 0
    target = target[to_keep]
    output = output[to_keep]

    # compute loss and backpropogate
    loss = criterion(output, target) # step 2
    
    # Clear the gradient buffers of the optimized parameters.
    # Otherwise, gradients from the previous batch would be accumulated.
    optimizer.zero_grad() # step 3
    
    loss.backward() # step 4
    
    optimizer.step() # step 5
    
    return model


# compoute correlation with held-out test data... 
def test_record_each_output(model, test_data, device, batch_size, n_sequences_test, gen_batch_data,n_back, choice_only=False, use_human_data=False):
    
    # Set the model to evaluation mode. This will turn off layers that would
    # otherwise behave differently during training, such as dropout. 
    model.eval()

    # number of batches
    n_batches = int(np.round(n_sequences_test/batch_size));

    # store output and target
    output_all = np.zeros((0,3))
    target_all = np.zeros((0,3))

    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we don't need the gradients at this point.
    with torch.no_grad():
        for batch_idx in range(n_batches):
            data, target = gen_batch_data(batch_size, batch_idx, test_data, use_human_data = use_human_data)
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            output = model(data)
            # Pick only the output corresponding to n_back
            if not choice_only:
                output = output[:, -n_back, :]
                target = target[:,-n_back,:]

            output_all = np.concatenate((output_all, output.cpu().numpy()))
            target_all = np.concatenate((target_all, target.cpu().numpy()))

    return (output_all, target_all)

# compute correlations by time-step back...
def compute_heldout_performance(trained_model, test_data, device, batch_size, n_sequences_test,gen_batch_data, n_back, choice_only=False, use_human_data=False):
    
    # also compute fixation and choice MSE...
    
    output_all, target_all = test_record_each_output(trained_model, test_data, device, batch_size, n_sequences_test,gen_batch_data, n_back,choice_only=choice_only, use_human_data=use_human_data)
    
    # compute the correlation
    output_flat = output_all.flatten()
    target_flat = target_all.flatten()
    output_flat = output_flat[target_flat != 0]
    target_flat = target_flat[target_flat != 0]
    
    this_corr = np.corrcoef(output_flat, target_flat)[1][0]
    
    # also compute summed squared error...
    this_mse = np.mean(np.power(output_flat - target_flat,2))
    this_nitems = len(output_flat)
    
    # compute pct correct max and min and order
    target_all_FILT = target_all[target_all[:,1] != 0, :]
    output_all_FILT = output_all[target_all[:,1] != 0, :]
    
    output_max_item = output_all_FILT.argmax(axis=1)
    target_max_item = target_all_FILT.argmax(axis=1)
    pct_correct_max = np.sum(output_max_item == target_max_item)/len(output_max_item)

    output_min_item = output_all_FILT.argmin(axis=1)
    target_min_item = target_all_FILT.argmin(axis=1)
    
    pct_correct_min = np.sum(output_min_item == target_min_item)/len(output_min_item)
    correct_order = (output_min_item == target_min_item) & (output_max_item == target_max_item)
    pct_correct_order = np.sum(correct_order)/len(correct_order)
    
    return this_corr, pct_correct_max, pct_correct_min, pct_correct_order, this_mse, this_nitems