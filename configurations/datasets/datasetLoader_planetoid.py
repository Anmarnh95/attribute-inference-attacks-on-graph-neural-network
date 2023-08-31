from configurations.datasets.datasetloaderinterface import DatasetLoaderInterface
from torch_geometric.datasets import Planetoid
import copy
import torch
import numpy as np
from logging import info as l
from logging import debug as d


class DatasetLoader_Planetoid(DatasetLoaderInterface):

    def __init__(self, dataset_name = "Cora", train_split = 0, test_split = 0):
        self.ds_name = dataset_name
        self.train = train_split
        self.test = test_split

        # If no train or no test split are given (one of them is zero), use public split
        if train_split == 0 or test_split == 0:
            self.ds = Planetoid(root="/tmp/PLANETOID", name=self.ds_name, split="public")
            return

        if dataset_name == "Cora":
            self.classes = 7
        elif dataset_name == "Pubmed":
            self.classes = 3
        else: 
            self.classes = 6

        train_per_class = int(train_split/self.classes)
        

        self.ds = Planetoid(root="/tmp/PLANETOID", name=self.ds_name, split="random", num_train_per_class=train_per_class,num_test=test_split,num_val=test_split)

        d(f"Number of Training nodes: {self.ds.train_mask.sum()}")
        d(f"Number of Testing nodes: {self.ds.test_mask.sum()}")

        # The number of training nodes in self.ds is not the same as in train_split, the following will complete the number of nodes
        rest_train = train_split - self.ds.data.train_mask.sum()

        if rest_train > 0:
            d(f"{rest_train} nodes need to be added to the trainig set to reach the desired {self.train} training nodes")
            
            indices_train = torch.nonzero((self.ds.train_mask == True), as_tuple=True)[0].numpy()
            indices_test = torch.nonzero((self.ds.test_mask == True), as_tuple=True)[0].numpy()

            # Choose nodes outisde of the train and test set
            choices = np.setdiff1d(range(self.ds.data.num_nodes),np.union1d(indices_train,indices_test))
            
            new_samples = np.random.choice(choices,rest_train.item(),replace=False)
            
            self.ds.data.train_mask[new_samples] = True

        d(f"Number of Training nodes: {self.ds.train_mask.sum()}")
        d(f"Number of Testing nodes: {self.ds.test_mask.sum()}")


    def get_data(self):
        return copy.deepcopy(self.ds.data)

