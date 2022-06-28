# functions for loading data, and generating batch data to train neural nets...
import os
import json
from res.sequential_tasks import pad_sequences, to_categorical
import numpy as np
import random

def load_data(sim_data_path, human_data_path,split_human_data=False, this_seed = 0):

    random.seed(this_seed)
    
    train_file_idxs = range(1,23)
    test_file_idxs = range(24,31)

    train_files = [os.path.join(sim_data_path, str(i) + '.json') for i in train_file_idxs]
    test_files = [os.path.join(sim_data_path, str(i) + '.json') for i in test_file_idxs]

    a = [json.load(open(train_files[i])) for i in range(15)]
    train_trials = [item for sublist in a for item in sublist]
    del a
    train_data_sim = train_trials[:int(1.5e6)]

    test_trials = json.load(open(test_files[0]))
    test_data_sim = test_trials[:int(1e5)]

    human_data = json.load(open(human_data_path))
    
    random.shuffle(train_data_sim)
    random.shuffle(test_data_sim)
    random.shuffle(human_data)
    
    if split_human_data:
        n_test=int(np.round(len(human_data)/3)) # test on 1/3
        human_test_data = human_data[:n_test]
        human_train_data = human_data[n_test+1:]

        return train_data_sim, test_data_sim, human_train_data, human_test_data
    else:
        return train_data_sim, test_data_sim, human_data

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


