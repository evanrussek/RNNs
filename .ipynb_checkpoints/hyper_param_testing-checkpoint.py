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
model_names = ['LSTM','GRU','Transformer']

### Train on these vals... 
hidden_sizes = np.array([64, 128, 256]) # 
sim_lrs = np.array([1e-3])

# what human lrs to use for training? -- try a bunch between 1e-5 and 1e-4...
human_lrs_train = np.array([1e-4, 1e-3])

# what human lrs to use for fine-tuning? # this might not be enough examples??
human_lrs_finetune = np.array([1e-4, 1e-3])

# finetune?

# For the transformer, also do this
transformer_attention_heads = [4,8]
transformer_layers = [2,4]
dropout = .2

# ...
# take an average over 3 runs...
n_runs = 3

# vary model name, hidden size, learning rate, (transformer stuff if transformer), run
job_model_names = []
job_runs = []
job_hidden_sizes = []
job_sim_lrs = []
job_human_lrs = []
job_attention_heads = []
job_layers = []
job_n_sim_seqs = []
job_n_human_seqs = []
job_dropout = []


# Build the train on Sim jobs...

# run this for 1 mil, and also on more jobs...
n_sim_seqs = 5e5
n_human_seqs = 0


for run in range(n_runs):
    for model_name in model_names:
        for hidden_size in hidden_sizes: # just run this for 512
            for sim_lr in sim_lrs:
                
                if model_name == 'Transformer':
                    for layer in transformer_layers:
                        for attention_heads in transformer_attention_heads:
                            
                            job_runs.append(run)
                            job_model_names.append(model_name)
                            job_hidden_sizes.append(hidden_size)
                            job_sim_lrs.append(sim_lr)
                            job_human_lrs.append(0)
                            
                            job_attention_heads.append(attention_heads)
                            job_layers.append(layer)
                            
                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
                            job_dropout.append(dropout)
                else:
                    job_runs.append(run)
                    job_model_names.append(model_name)
                    job_hidden_sizes.append(hidden_size)
                    job_sim_lrs.append(sim_lr)
                    job_human_lrs.append(0)
                    
                    job_attention_heads.append(0)
                    job_layers.append(0)
                    
                    job_n_sim_seqs.append(n_sim_seqs)
                    job_n_human_seqs.append(n_human_seqs)
                    job_dropout.append(dropout)
                    

                    
# try using smaller model sizes for this?
# Add the train on human jobs alone... RERUN THESE!!!
n_sim_seqs = 0
n_human_seqs = 1e6

for run in range(n_runs):
    for model_name in model_names:
        for hidden_size in hidden_sizes:
             for human_lr in human_lrs_train:
                if model_name == 'Transformer':
                    for layer in transformer_layers:
                        for attention_heads in transformer_attention_heads:
                            
                            job_runs.append(run)
                            job_model_names.append(model_name)
                            job_hidden_sizes.append(hidden_size)
                            job_sim_lrs.append(0)
                            job_human_lrs.append(human_lr)

                            job_attention_heads.append(attention_heads)
                            job_layers.append(layer)
                            
                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
                            job_dropout.append(dropout)

                else:
                    job_runs.append(run)
                    job_model_names.append(model_name)
                    job_hidden_sizes.append(hidden_size)
                    job_sim_lrs.append(0)
                    job_human_lrs.append(human_lr)

                    job_attention_heads.append(0)
                    job_layers.append(0)
                    
                    job_n_sim_seqs.append(n_sim_seqs)
                    job_n_human_seqs.append(n_human_seqs)
                    job_dropout.append(dropout)

                    

# Add the finetune on human jobs... -- no learning rate for the human simulation here...
n_sim_seqs = 3e5 # what's the good pretrain val?
n_human_seqs = 1e6

for run in range(n_runs):
    for model_name in model_names:
        for hidden_size in hidden_sizes:
             for human_lr in human_lrs_finetune:
                if model_name == 'Transformer':
                    for layer in transformer_layers:
                        for attention_heads in transformer_attention_heads:
                            
                            job_runs.append(run)
                            job_model_names.append(model_name)
                            job_hidden_sizes.append(hidden_size)
                            job_sim_lrs.append(.001)
                            job_human_lrs.append(human_lr)

                            job_attention_heads.append(attention_heads)
                            job_layers.append(layer)
                            
                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
                            
                            job_dropout.append(dropout)

                else:
                    job_runs.append(run)
                    job_model_names.append(model_name)
                    job_hidden_sizes.append(hidden_size)
                    job_sim_lrs.append(.001) # just use a neutral sim learning rate...
                    job_human_lrs.append(human_lr)

                    job_attention_heads.append(0)
                    job_layers.append(0)
                    
                    job_n_sim_seqs.append(n_sim_seqs)
                    job_n_human_seqs.append(n_human_seqs)
                    
                    job_dropout.append(dropout)

                    

# Add the train on human and fine-tune on sim jobs... 
# what's the file name
run_idx = job_runs[job_idx]
model_name = job_model_names[job_idx]
d_model = job_hidden_sizes[job_idx]
sim_lr = job_sim_lrs[job_idx]
human_lr = job_human_lrs[job_idx]
n_head = job_attention_heads[job_idx]
n_layers = job_layers[job_idx]
n_simulation_sequences_train = job_n_sim_seqs[job_idx]
n_human_sequences_train = job_n_human_seqs[job_idx]
dropout = job_dropout[job_idx]

"""
if dropout>0:
    save_file_name = 'run_{}_model_name_{}_d_model_{}_sim_lr_{}_human_lr_{}_n_head_{}_n_layers_{}_nsim_{}_nhum_{}_do_{}_fu_all'.format(run_idx, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_simulation_sequences_train, n_human_sequences_train, dropout)
else:
    save_file_name = 'run_{}_model_name_{}_d_model_{}_sim_lr_{}_human_lr_{}_n_head_{}_n_layers_{}_nsim_{}_nhum_{}_fu_all'.format(run_idx, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_simulation_sequences_train, n_human_sequences_train)
"""
    
if dropout>0:
    save_file_name = 'run_{}_model_name_{}_d_model_{}_sim_lr_{}_human_lr_{}_n_head_{}_n_layers_{}_nsim_{}_nhum_{}_do_{}_fu_all_fix_and_choice'.format(run_idx, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_simulation_sequences_train, n_human_sequences_train, dropout)
else:
    save_file_name = 'run_{}_model_name_{}_d_model_{}_sim_lr_{}_human_lr_{}_n_head_{}_n_layers_{}_nsim_{}_nhum_{}_fu_all_fix_and_choice'.format(run_idx, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_simulation_sequences_train, n_human_sequences_train)
 

# Call key function  -- 5e5 sequences?
main_as_fun(
    run_idx = job_runs[job_idx],
    model_name = job_model_names[job_idx],
    d_model = job_hidden_sizes[job_idx],
    sim_lr = job_sim_lrs[job_idx],
    human_lr = job_human_lrs[job_idx],
    n_head = job_attention_heads[job_idx],
    n_layers = job_layers[job_idx],
    train_seq_part = 'fix_and_choice',
    fix_unit = 'all',
    n_simulation_sequences_train = job_n_sim_seqs[job_idx],
    n_human_sequences_train = job_n_human_seqs[job_idx],
    dropout = job_dropout[job_idx],
    on_cluster = True,
    save_folder_name = 'Hyper_Param_Search2',
    save_file_name = save_file_name)