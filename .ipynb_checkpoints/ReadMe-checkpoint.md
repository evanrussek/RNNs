Code for using neural networks to both invert an optimal model of human fixations, and then predicting  human utilities over food items from fixations and choices.

Please email Evan Russek, evrussek@gmail.com, with any questions.

Note that much of this code borrows heavily from from NYU deep learning course, https://atcold.github.io/pytorch-Deep-Learning/

Brief Description of key python files:

main_as_fun.py takes in a number of parameters, trains a model on some number of simulation data and human data, and then tests the model on simulation and human data

for running on the cluster and calling many types of paramaterizations, RNN_cpu_parallel.slurm runs an array job
it calls pref_master_file.py which takes the array index and then runs one set of parameters.

The following python files define a number of classes and functions that are called by a number of scripts

neural_nets.py: defines LSTM and MLP pytorch neural networks. 

sequential_tasks.py: taken from nyu deep learning course, defines some useful function for augmenting data (e.g. padding and making seqeunces).

load_data_funs.py: defines a number of functions useful for loading in either simulated or human fixation data, defining train vs test sets, and then also functions which define unique batches.

AnalyzeResults notebook makes all plots in the writeup, using the results from the cluster runs.
