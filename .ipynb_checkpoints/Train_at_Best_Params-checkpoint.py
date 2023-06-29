import os
import numpy as np
import pickle

from main_as_fun import main_as_fun


is_array_job=True

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 3

## Make folder to save best params
# best_param_folder = '/home/erussek/projects/RNNs/best_hyper_params'
best_param_folder = '/home/erussek/projects/RNNs/best_hyper_params_fix_only'

# Read in best hyper params...
best_param_dict = pickle.load(open(os.path.join(best_param_folder, 'best_hyper_params.pkl'), 'rb'))

# types of models...
model_names = ['LSTM', 'GRU', 'Transformer']

train_names = ['train_sim_test_sim', 'train_sim_test_human', 'train_human_test_human', 'train_sim_human_test_human']


job_model_names = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_attention_heads = []
job_layers = []
job_runs = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_train_names = []
job_dropout = []

n_runs = 20
for run in range(n_runs):
    for tn in train_names:
        for model_name in model_names:
            mbp = best_param_dict[tn][model_name]

            job_runs.append(run)
            job_model_names.append(model_name)
            job_hidden_sizes.append(mbp['d_model'])
            job_sim_lrs.append(mbp['sim_lr'])
            job_human_lrs.append(mbp['human_lr'])
            job_attention_heads.append(mbp['n_head'])
            job_layers.append(mbp['n_layers'])
            job_n_sim_seqs.append(mbp['best_sim_num'])
            job_n_human_seqs.append(mbp['best_hum_num'])
            job_train_names.append(tn)
            job_dropout.append(mbp['dropout'])


run_idx = job_runs[job_idx]
model_name = job_model_names[job_idx]
d_model = job_hidden_sizes[job_idx]
sim_lr = job_sim_lrs[job_idx]
human_lr = job_human_lrs[job_idx]
n_head = job_attention_heads[job_idx]
n_layers = job_layers[job_idx]
n_simulation_sequences_train = job_n_sim_seqs[job_idx]
n_human_sequences_train = job_n_human_seqs[job_idx]
train_name = job_train_names[job_idx]
dropout = job_dropout[job_idx]

save_file_name = 'run_{}_model_name_{}_train_name_{}_fix_and_sim_bp_fix_only'.format(run_idx, model_name, train_name)


# Call key function  -- 5e5 sequences?
main_as_fun(
    run_idx = job_runs[job_idx],
    model_name = job_model_names[job_idx],
    d_model = job_hidden_sizes[job_idx],
    sim_lr = job_sim_lrs[job_idx],
    human_lr = job_human_lrs[job_idx],
    n_head = job_attention_heads[job_idx],
    n_layers = job_layers[job_idx],
    train_seq_part = 'fix_only',
    n_simulation_sequences_train = job_n_sim_seqs[job_idx],
    n_human_sequences_train = job_n_human_seqs[job_idx],
    dropout = job_dropout[job_idx],
    on_cluster = True,
    save_folder_name = 'Best_Param_Results',
    save_file_name = save_file_name)