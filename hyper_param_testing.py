# run through diff hyper parameters for training on sim (and testing on human)

import os
import numpy as np

from main_as_fun import main_as_fun

is_array_job=False

if is_array_job:
    job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
else:
    job_idx = 0


# loop through:
# model type, hidden size, learning rate (sim data), transformer attention heads, trans. layers, runs

# types of models...
model_names = ['LSTM', 'GRU', 'Transformer']

### Train on these vals... 
hidden_sizes = np.array([32, 64, 128])
sim_lrs = np.array([1e-4, 1e-3, 1e-2])

# what human lrs to use for training?
human_lrs_train = np.array([1e-4, 1e-3, 1e-2])

# what human lrs to use for fine-tuning?
human_lrs_finetune = np.array([1e-5, 1e-4, 1e-3])


# finetune?

# For the transformer, also do this
transformer_attention_heads = [4,8]
transformer_layers = [2,4]

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
job_runs = []
job_n_sim_seqs = []
job_n_human_seqs = []



# Build the train on Sim jobs...
n_sim_seqs = 1e3#5e5
n_human_seqs = 0

for run in range(n_runs):
    for model_name in model_names:
        for hidden_size in hidden_sizes:
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
                            job_layers.append(job_layers)
                            
                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
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
                    


# Add the train on human jobs alone...
n_sim_seqs = 0
n_human_seqs = 5e5 

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
                            job_layers.append(job_layers)
                            
                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
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
                    
                    

# Add the finetune on human jobs...
n_sim_seqs = 3e5 # what's the good pretrain val?
n_human_seqs = 5e5 

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
                            job_sim_lrs.append(0)
                            job_human_lrs.append(human_lr)

                            job_attention_heads.append(attention_heads)
                            job_layers.append(job_layers)
                            
                            job_n_sim_seqs.append(n_sim_seqs)
                            job_n_human_seqs.append(n_human_seqs)
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

save_file_name = 'run_{}_model_name_{}_d_model_{}_sim_lr_{}_human_lr_{}_n_head_{}_n_layers_{}_nsim_{}_nhum_{}'.format(run_idx, model_name, d_model, sim_lr, human_lr, n_head, n_layers, n_simulation_sequences_train, n_human_sequences_train)

# Call key function  -- 5e5 sequences?
main_as_fun(
    run_idx = job_runs[job_idx],
    model_name = job_model_names[job_idx],
    d_model = job_hidden_sizes[job_idx],
    sim_lr = job_sim_lrs[job_idx],
    human_lr = job_human_lrs[job_idx],
    n_head = job_attention_heads[job_idx],
    n_layers = job_layers[job_idx],
    
    n_simulation_sequences_train = job_n_sim_seqs[job_idx],
    n_human_sequences_train = job_n_human_seqs[job_idx],

    on_cluster = False,
    save_folder_name = 'Scratch_Hyper_Param',
    save_file_name = save_file_name)