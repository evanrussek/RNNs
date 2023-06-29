# run through diff hyper parameters for training on sim (and testing on human)

import os
import numpy as np

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 0

run_idx = job_idx
to_save_folder = '/scratch/gpfs/erussek/RNN_project/Best_Param_Results'

n_sim_seqs = 1e5
n_human_seqs = 0

train_name = 'train_sim_test_sim'
save_file_name = 'run_{}_train_name_{}_choice_then_fix'.format(run_idx, train_name)

main_as_fun(
    run_idx = run_idx,
    model_name = 'MLP',
    train_seq_part = 'choice_then_fix',
    d_model = 16,
    sim_lr = .001,
    human_lr = .001,
    n_simulation_sequences_train = 1e5,
    n_human_sequences_train = 0,
    dropout = 0,
    on_cluster = True,
    save_folder_name = 'Best_Param_Results',
    save_file_name = save_file_name)

train_name = 'train_human_test_human'
save_file_name = 'run_{}_train_name_{}_choice_then_fix'.format(run_idx, train_name)

main_as_fun(
    run_idx = run_idx,
    model_name = 'MLP',
    train_seq_part = 'choice_then_fix',
    d_model = 16,
    sim_lr = .001,
    human_lr = .001,
    n_simulation_sequences_train = 0,
    n_human_sequences_train = 1e5,
    dropout = 0,
    on_cluster = True,
    save_folder_name = 'Best_Param_Results',
    save_file_name = save_file_name)

train_name = 'train_sim_human_test_human'
save_file_name = 'run_{}_train_name_{}_choice_then_fix'.format(run_idx, train_name)

main_as_fun(
    run_idx = run_idx,
    model_name = 'MLP',
    train_seq_part = 'choice_then_fix',
    d_model = 16,
    sim_lr = .001,
    human_lr = .001,
    n_simulation_sequences_train = 1e5,
    n_human_sequences_train = 1e5,
    dropout = 0,
    on_cluster = True,
    save_folder_name = 'Best_Param_Results',
    save_file_name = save_file_name)