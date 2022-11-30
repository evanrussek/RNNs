# master file that takes in cluster array job and assigns it to correct paramaterization for call to train and test model

import os
import numpy as np

from main_as_fun import main_as_fun

is_array_job=True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 0

seq_model_names = ["LSTM", "GRU", "Transformer"]
train_seq_parts = ["fix_and_choice", "fix_only", "choice_only"]

job_model_names = []
job_seq_parts = []
job_run_idxs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_lr_human = []

n_sim_seqs = np.array([0., 3e5], dtype=int)
n_human_seqs = np.array([108800, 179200], dtype=int) # check this...
human_lr = np.array([1e-3, 5e-5])

n_runs_per_setting = 10 # 20 runs

for ss_idx in range(len(n_sim_seqs)):
    sim_seqs = n_sim_seqs[ss_idx]
    human_seqs = n_human_seqs[ss_idx]
    for run_idx in range(n_runs_per_setting):
        for seq_part in train_seq_parts:
            if seq_part != "choice_only":
                for model_name in seq_model_names:            
                    job_model_names.append(model_name)
                    job_seq_parts.append(seq_part)
                    job_n_sim_seqs.append(sim_seqs)
                    job_n_human_seqs.append(human_seqs)
                    job_run_idxs.append(run_idx)
                    job_lr_human.append(human_lr[ss_idx])

            else:
                job_seq_parts.append(seq_part)
                job_model_names.append('MLP')
                job_run_idxs.append(run_idx)
                job_n_sim_seqs.append(sim_seqs)
                job_n_human_seqs.append(human_seqs)
                job_lr_human.append(human_lr[ss_idx])




# Call key function !
main_as_fun(model_name=job_model_names[job_idx], train_seq_part = job_seq_parts[job_idx], n_simulation_sequences_train = job_n_sim_seqs[job_idx], n_human_sequences_train = job_n_human_seqs[job_idx], run_idx=job_run_idxs[job_idx], human_lr = job_lr_human[job_idx], test_batch_increment_sim = 5e5, test_batch_increment_human = 100) # test human every 100 iters?




