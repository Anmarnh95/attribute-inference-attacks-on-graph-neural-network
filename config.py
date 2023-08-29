from easydict import EasyDict as edict
import numpy as np
from registeration import *
import logging

config = edict()

# sets the desired logging level
config.log_level = logging.DEBUG

# sets the initial random seed
config.random_seed = 32

#__________________________________________________________________________
# TARGET MODEL PARAMETERS

# Size of hidden layer in nonprivate model
config.hidden_size = 10

# Number of training epochs
config.epochs = 500

# Learning rate of GraphSAGE
config.lr = 0.01

# Weight Decay of GraphSAGE
config.wd = 5e-4

# Dropout rate
config.dropout = 0.5

#__________________________________________________________________________
# PRIVATE MODEL SETTINGS
# OPTIONALS

'''
Every model defines it's own privacy parameters, here, the privacy parameters needs to be defined as def and the
values of each of these parameters must be given as a list. 
For example, LPGNN has two epsilons, eps_x and eps_y, and two KProp parameters, Kx and Ky, they should look as the following: 
    config.privacy_parameters = {"eps_x": [np.inf, 3, 1, 0.01], 
                                "eps_y": [np.inf, 3, 1, 0.01],
                                "Kx": [0],
                                "Ky": [0]}
Another example is PrivGNN, which uses lambda instead of epsilon. This could be represented as follows:
    config.privacy_parameters = {"lambda": [0.2, 0.4, 0.6, 1]}

Give an empty dict if you don't require any privacy.
'''
# config.privacy_parameters = {"eps_x": [np.inf,3,1,0.01], 
#                                 "eps_y": [np.inf,3,1,0.01],
#                                 "Kx": [0],
#                                 "Ky": [0]}
config.privacy_parameters = {}

#__________________________________________________________________________
# Attack SETTINGS

'''
The name of the model used. GraphSAGE or GCN for nonprivate model, LPGNN for private one.

Define the target model using the interface TargetModelInterface in configurations\models. 
1. Make a new file with the name "model_{modelname}.py in the directory configurations\models"
2. In the file, define a class with inherits targetmodelinterface and implements the necessary functions
3. Add model to Registerations.py in the function return_target_model

Currently, the available target models are: GCN, SAGE, MLP.
'''

config.model_name = "GCN"

# Confidence score threshold used for MA.
config.MA_threshold = 0.7

'''
List of variable settings K for attack's KNN runs! For each entry, all specified will attack 10 times for each RAA and SAA.
If k_list includes 0, attacker will have full graph acces. 
'''
config.k_list = [0, 2, 5, 8]

# If true, will run BF attack with one missing attribute only.
# WARNING: bf attack is not yet up to date.. it might not always work..
config.bf_included = False

# If true, will run MA attack.
config.ma_included = False

# If true, will run FP attack.
config.fp_included = False

# If true, will run random initializer attack
config.ri_included = True

# Ig true, will run MA with random initializer as core instead of FP
config.rima_included = False

# Iterations of the feature Propagation
config.fp_iter = 10

# Run numbers
config.run_numbers = range(1,11)

#__________________________________________________________________________
# Dataset SETTINGS

'''
The name of the dataset used. Currently, planetoid is implemented, meaning that Cora, Pubmed and CiteSeer are available.

To make a dataset available, the loader of the dataset must be implemented. To do that,
define the loader using the interface DatasetLoaderInterface in configurations\datasets. 
1. Make a new file with the name "datasetloader_{loadername}.py in the directory configurations\models"
2. In the file, define a class which inherits datasetloaderinterface and implements the necessary functions
3. Add the datasets you want to use to registerations.py in the function return_dataset_loader

Currently, the available datasets are: Facebook, LastFM, Cora, Pubmed, CiteSeer, Credit, Texas100X, Test
'''

config.dataset_name = "Cora"

# The sensetive attributes are always missing when attacking.
#config.sensetive_attr = [1,12]
#config.sensetive_attr = [1,100]

config.sensetive_attr = [1]

# If true, will consider the defined dataset to be binary.
if config.dataset_name in ['Cora','Facebook','Credit','Texas100X','Test']:
    config.binary = True
else: 
    config.binary = False

# limits the inference of FP and RI to a lower and an upper limit. First entry in the tupel is the min value while the second one is the maximum value.
if config.binary == False:
    config.min_max_dataset_values = (-1,16)
else:
    config.min_max_dataset_values = (0,1)

# This will round the values in feature propogation while attacking, removing all floating point values.
config.round = False

# The code will always run the experiment with SAA, but if this variable is set to true, it will also run it for RAA
config.RAA = False

# Training to test split in the form (train/test). Setting either one to zero will indicate that the public split should 
# be used
config.split = (500,100)


# Dectates how many of the candidates set's sensetive attributes are going to be perturbed
config.perturbation_ratio = [1, 0.5]

# Number of candidates in the candidate set. Experiment run will be repeated for each of the given number of candidates. 
#config.candidate_set_list = [2]
config.candidate_set_list = [100]

# Given values for m. The experiment will be repeated for the given values of m. A value of 0.5 is 50% and it is the default value.
# the parameter m states the amount of attributes which will be made missing. A 100% means that ALL the attributes of a node is made missing.
config.m_list = [0]


#__________________________________________________________________________
# Shadow Attack SETTINGS

config.run_shadow_attack = False

config.target_to_shadow_rate = 0.5

config.test_rate = 0.5

config.shadow_model_name = "GCN"

config.shadow_perturbation_ratio = [0.5]

config.shadow_debug = True


