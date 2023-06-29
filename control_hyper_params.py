# run through diff hyper parameters for training on sim (and testing on human)

import os
import numpy as np

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 10

# loop through:

# model type, hidden size, learning rate (sim data), transformer attention heads, trans. layers, runs

# types of models...

### Train on these vals... 
hidden_sizes = np.array([32]) 
sim_lrs = np.array([1e-3])

# what human lrs to use for training?
human_lrs_train = np.array([5e-4])

# what human lrs to use for fine-tuning? # this might not be enough examples??
human_lrs_finetune = np.array([5e-4])
dropout_vals = [0]

# fixation types
fix_units = ['ID', 'sum', 'prop']
fix_units = ['all']

# vary model name, hidden size, learning rate, (transformer stuff if transformer), run
job_runs = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_dropout = []
job_fix_units = []


# run this for 1 mil, and also on more jobs...
n_sim_seqs = 5e5
n_human_seqs = 0

n_runs = 20

for run in range(n_runs):
    for fu in fix_units:        
        for hidden_size in hidden_sizes: # just run this for 512
            for sim_lr in sim_lrs:
                for dropout in dropout_vals:
                    job_runs.append(run)
                    job_fix_units.append(fu)
                    job_hidden_sizes.append(hidden_size)
                    job_sim_lrs.append(sim_lr)
                    job_human_lrs.append(0)
                    job_n_sim_seqs.append(n_sim_seqs)
                    job_n_human_seqs.append(n_human_seqs)
                    job_dropout.append(dropout)
                    

# try using smaller model sizes for this?
# Add the train on human jobs alone... RERUN THESE!!!
n_sim_seqs = 0
n_human_seqs = 5e5

for run in range(n_runs):
    for fu in fix_units:        
        for hidden_size in hidden_sizes: # just run this for 512
            for human_lr in human_lrs_finetune:
                for dropout in dropout_vals:
                    job_runs.append(run)
                    job_fix_units.append(fu)
                    job_hidden_sizes.append(hidden_size)
                    job_sim_lrs.append(0)
                    job_human_lrs.append(human_lr)
                    job_n_sim_seqs.append(n_sim_seqs)
                    job_n_human_seqs.append(n_human_seqs)
                    job_dropout.append(dropout)
                    
                    
# Add the finetune on human jobs... -- no learning rate for the human simulation here...
n_sim_seqs = 3e5 # what's the good pretrain val?
n_human_seqs = 5e5

for run in range(n_runs):
    for fu in ['all']:#fix_units:        
        for hidden_size in hidden_sizes: # just run this for 512
            for human_lr in human_lrs_train:
                for dropout in dropout_vals:
                    job_runs.append(run)
                    job_fix_units.append(fu)
                    job_hidden_sizes.append(hidden_size)
                    job_sim_lrs.append(.001)
                    job_human_lrs.append(human_lr)
                    job_n_sim_seqs.append(n_sim_seqs)
                    job_n_human_seqs.append(n_human_seqs)
                    job_dropout.append(dropout)


# Add the train on human and fine-tune on sim jobs... 
# what's the file name
run_idx = job_runs[job_idx]
d_model = job_hidden_sizes[job_idx]
sim_lr = job_sim_lrs[job_idx]
human_lr = job_human_lrs[job_idx]
n_simulation_sequences_train = job_n_sim_seqs[job_idx]
n_human_sequences_train = job_n_human_seqs[job_idx]
dropout = job_dropout[job_idx]
fix_unit = job_fix_units[job_idx]

save_file_name = 'run_{}_fixunit_{}_d_model_{}_sim_lr_{}_human_lr_{}_nsim_{}_nhum_{}_do_{}'.format(run_idx, fix_unit, d_model, sim_lr, human_lr, n_simulation_sequences_train, n_human_sequences_train, dropout)

# Call key function  -- 5e5 sequences?

# do it for the LSTM...

main_as_fun(
    run_idx = job_runs[job_idx],
    fix_unit = job_fix_units[job_idx],
    model_name = 'MLP',
    train_seq_part = 'fix_only',
    d_model = job_hidden_sizes[job_idx],
    sim_lr = job_sim_lrs[job_idx],
    human_lr = job_human_lrs[job_idx],
    n_simulation_sequences_train = job_n_sim_seqs[job_idx],
    n_human_sequences_train = job_n_human_seqs[job_idx],
    dropout = job_dropout[job_idx],
    on_cluster = True,
    save_folder_name = 'Control_Models1_HP',
    save_file_name = save_file_name)