import os
import numpy as np
import pickle

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    this_job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    this_job_idx = 3
    
## Make folder to save best params
best_param_folder = '/home/erussek/projects/RNNs/best_hyper_params_choice_only'

# Read in best hyper params...
best_param_dict = pickle.load(open(os.path.join(best_param_folder, 'best_hyper_params.pkl'), 'rb'))

train_seq_parts = ["choice_only", "choice_then_fix"]
train_seq_parts = ["choice_only"]

part_names = ["Simulated_Only", "Human_Only", "Simulated_and_Human"]

job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_runs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_train_names = []
job_full_names = []
job_tsps = []

n_runs = 200
for run in range(n_runs):
    for tsp in train_seq_parts:
        for part_name in part_names:
            
            full_name = "{}_{}".format(tsp, part_name)
            job_full_names.append(full_name + '_run_{}'.format(run))
            mbp = best_param_dict[full_name]
            
            job_runs.append(run)
            job_tsps.append(tsp)
            job_hidden_sizes.append(mbp['d_model'])
            job_sim_lrs.append(mbp['sim_lr'])
            job_human_lrs.append(mbp['human_lr'])
            
                                
            if part_name == 'Simulated_Only':
                job_n_sim_seqs.append(1e6)
            else:
                job_n_sim_seqs.append(mbp['best_sim_num'])
                
            job_n_human_seqs.append(mbp['best_hum_num'])
            job_train_names.append(part_name)

# run 400 jobs...
which_jobs = np.reshape(np.arange(600), (int(600/3),3))
these_jobs = which_jobs[this_job_idx,:]

for job_idx in these_jobs:
    
    print(job_idx)
    

    save_file_name = job_full_names[job_idx]

    main_as_fun(
        run_idx = job_runs[job_idx],
        model_name = 'MLP',
        d_model = job_hidden_sizes[job_idx],
        sim_lr = job_sim_lrs[job_idx],
        human_lr = job_human_lrs[job_idx],
        train_seq_part = job_tsps[job_idx],
        n_simulation_sequences_train = job_n_sim_seqs[job_idx],
        n_human_sequences_train = job_n_human_seqs[job_idx],
        dropout = .1,
        on_cluster = True,
        save_folder_name = 'Choice_Only_Best_Param_Results',
        save_file_name = save_file_name)