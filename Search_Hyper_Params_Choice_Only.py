# run through diff hyper parameters for training on sim (and testing on human)

import os
import numpy as np

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 10

### Train on these vals... 
hidden_sizes = np.array([32, 64, 128, 256])
sim_lrs = np.array([1e-4, 1e-3])
human_lrs_train = np.array([1e-4, 1e-3])
human_lrs_finetune = np.array([1e-5, 1e-4, 1e-3])
dropout_vals = [.1]
train_seq_parts = ["choice_only", "choice_then_fix"]

job_runs = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_part_names = []
job_tsps = []


n_sim_seqs = 5e5
n_human_seqs = 0
part_name = "Simulated_Only"

n_runs = 2

for run in range(n_runs):
    for hidden_size in hidden_sizes: # just run this for 512
        for sim_lr in sim_lrs:
            for tsp in train_seq_parts: 
                job_runs.append(run)
                job_hidden_sizes.append(hidden_size)
                job_sim_lrs.append(sim_lr)
                job_human_lrs.append(0)
                job_n_sim_seqs.append(n_sim_seqs)
                job_n_human_seqs.append(n_human_seqs)
                job_part_names.append(part_name)
                job_tsps.append(tsp)

n_sim_seqs = 0
n_human_seqs = 1e6
part_name = "Human_Only"

for run in range(n_runs):
    for hidden_size in hidden_sizes: # just run this for 512
        for human_lr in human_lrs_train:
            for tsp in train_seq_parts: 
                job_runs.append(run)
                job_hidden_sizes.append(hidden_size)
                job_sim_lrs.append(0)
                job_human_lrs.append(human_lr)
                job_n_sim_seqs.append(n_sim_seqs)
                job_n_human_seqs.append(n_human_seqs)
                job_part_names.append(part_name)
                job_tsps.append(tsp)

                    

n_sim_seqs = 3e5 # what's the good pretrain val?
n_human_seqs = 1e6
part_name = "Simulated_and_Human"

for run in range(n_runs):
    for hidden_size in hidden_sizes:
        for human_lr in human_lrs_finetune:
            for tsp in train_seq_parts: 
                job_runs.append(run)
                job_hidden_sizes.append(hidden_size)
                job_sim_lrs.append(.001)
                job_human_lrs.append(human_lr)
                job_n_sim_seqs.append(n_sim_seqs)
                job_n_human_seqs.append(n_human_seqs)
                job_part_names.append(part_name)
                job_tsps.append(tsp)

                    
run_idx = job_runs[job_idx]
d_model = job_hidden_sizes[job_idx]
sim_lr = job_sim_lrs[job_idx]
human_lr = job_human_lrs[job_idx]
n_simulation_sequences_train = job_n_sim_seqs[job_idx]
n_human_sequences_train = job_n_human_seqs[job_idx]
part_name = job_part_names[job_idx]
tsp = job_tsps[job_idx]


save_file_name = '{}_run_{}_d_model_{}_sim_lr_{}_human_lr_{}_tsp_{}'.format(part_name,run_idx, d_model, sim_lr, human_lr, tsp)
                    
main_as_fun(
    run_idx = run_idx,
    model_name = 'MLP',
    train_seq_part = tsp,
    d_model = d_model,
    sim_lr = sim_lr,
    human_lr = human_lr,
    n_simulation_sequences_train = n_simulation_sequences_train,
    n_human_sequences_train = n_human_sequences_train,
    dropout = .1,
    on_cluster = True,
    save_folder_name = 'Hyper_Param_Search_Choice_Only',
    save_file_name = save_file_name)