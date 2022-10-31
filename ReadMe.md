This code reflects ongoing progress on attempting to use neural networks to both invert an optimal model of human fixations, and then tries to apply this to predict human utilities over food items from fixations and choices.

Please email Evan Russek, evrussek@gmail.com, with any questions.

Note that much of this code borrows heavily from from NYU deep learning course, https://atcold.github.io/pytorch-Deep-Learning/

Brief Description of key python files:

The following python files define a number of classes and functions that are called by a number of scripts

neural_nets.py: defines LSTM and MLP pytorch neural networks. 

sequential_tasks.py: taken from nyu deep learning course, defines some useful function for augmenting data (e.g. padding and making seqeunces).

load_data_funs.py: defines a number of functions useful for loading in either simulated or human fixation data, defining train vs test sets, and then also functions which define unique batches.

The following scripts carry out key stages of analysis (and are built to be called in turn by slurm scripts and run on princeton clusters):

note -- probably want to also find the best parameters for all of these?

find_optimal_params_for_train_sim_test_sim.py: uses the Optuna package to identify hyperparameters which maximize performance at training on simulated data, and predicting held-out simulated data. To run on cluster, with GPU, param_search_job.slurm is called.

train_sim_test_sim_and_human_w_optimal_params.py: trains neural networks using previously identified best hyperparameters on simulated data. generates analysis of test performance on both simualted and human data. To run on cluster (cpu though in parallel), use RNN_cpu_parallel.slurm. plots of this analysis are in the following notebooks: analyze_performance_of_trained_networks_w_good_params.ipynb (makes most plots in write up) and analyze_prediction_orderings.ipynb (examines propportion of times correct max vs min utility item is selected).

find_n_train_epochs_for_pretrain_sim_human_test_human.py: pretrains neural network on both simualted data (for 5 levels between 0 and 1.5 million trials) and 500 epochs of the human train set (looping through the total set many times) of human data, tests on held-out human data. Note that this is used to identify the number of pretraining examples, and also the number of epochs to train. Analysis of these results (with plots to choose the number of examples) is in the notebook: analyze_train_sim_human_test_human_select_n_train_epochs.ipynb

train_sim_and_human_test_human_w_optimal_train_epochs.py: uses above identified number of pretraining, as well as number of human epochs to train. then tests on human data. results are analyzed in analyze_performance_of_trained_networks_w_good_params.ipynb


