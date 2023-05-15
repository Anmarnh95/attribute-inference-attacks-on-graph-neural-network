# Attribute Inference Attack on Node Classification Tasks

This part holds the code for the experiment of attacking private models with attribute inference attack on node classification tasks. **the code here is still not dynamic enough and still needs improvments and arrangment.**

## Structure
* [runner](runner.py): The starting point for the experiment
* [config](config.py): This file has the configurations for the experiment, from the model to the dataset and the other variables used in the experiment.
* [models](configurations/models/) A directory which holds the implementations of the available target models.
* [datasets](configurations/datasets/) A directory which holds the implementations of the loaders, which load the available datasets.

## Running an Experiment

After specifing the configuartion and the settings of the experiment in [config](config.py), run the following: 

~~~bash
python runner.py
~~~

This will output two directories, a directory called **resultsOfRunner{model_name}{dataset_name}{training_split}** which holds the raw tensor outputs of the different attacks, which allows the user to check the outputs in details. The model_name, dataset_name, training_split are the same ones specified in [config](config.py). The second directory called **resultsOfRunner{model_name}{dataset_name}{training_split}_results** holds two json files, the first one has the target model accuracy, while the second one holds the final evaluation results after measuring the distances for each attacker. 

# configurating the config file

The [config](config.py) file gives you a lot of freedom to run the experiment, but this could be overwhelming at first. To make things simplier, all the settings of the config file should stay as they are, but only change the following: 

~~~python
config.dataset_name = "Cora"
~~~
This will specifiy which dataset you want to use, the available datasets are: Facebook, LastFM, Cora, Pubmed, CiteSeer, Credit, and Test. Test datasets is a dataset made for debugging the code. To add your own dataset, read the specified section in [config](config.py). The default dataset is Cora.

~~~python
config.model_name = "GCN"
~~~
This will specifiy which target model you want to use, the available models are: GCN, SAGE, and LPGNN. LPGNN needs more configurations compared to the first two, because its differentially private and has to be cloned or installed first. To run LPGNN, read [it's interface](configurations/models/model_sage.py) and the [config](config.py) carefully. 

To add your own model, read [config](config.py) carefully. The default model is GCN.

~~~python
config.random_seed = 32
~~~

This will specify the random seed.

~~~python
config.ma_included = True
config.fp_included = True
config.ri_included = True
config.rima_included = True
~~~

These will specify which attack you want to run during the experiment, which are MA, FP, RI, RIMA respectfuly. By default, they are all set to True, meaning that they will all run during an experiment run. To deactivate one of the attacks, set it to False.

~~~python
config.sensetive_attr = [1]
~~~

This will specify which sensetive attributes will be perturbed. The enteries in this list should be indexes of these attributes.

~~~python
config.min_max_dataset_values = (-1,16)
~~~

This will limits the inference of FP and RI to a lower and an upper limit. First entry in the tupel is the min value while the second one is the maximum value. This is only necessary for nonbinary datasets, where FP and RI could go overboard with their inference.

~~~python
config.split = (100,10)
~~~

This will specify the training and testing split of the experiment. First entery is the training split and the second one is the test split.

~~~python
config.candidate_set_list = [100]
~~~
This will specify Number of candidates in the candidate set. 