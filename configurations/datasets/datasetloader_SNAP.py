from configurations.datasets.datasetloaderinterface import DatasetLoaderInterface
from torch_geometric.datasets import SNAPDataset
import copy
import torch
import numpy as np
import torch_geometric.transforms as T
from logging import info as l
from logging import debug as d

class DatasetLoader_SNAP(DatasetLoaderInterface):

    def __init__(self, dataset_name = "Facebook", train_split = 0, test_split = 0):
        self.ds_name = dataset_name
        self.train = train_split
        self.test = test_split

        if dataset_name == "Facebook":
            ds_key = "ego-facebook"
        else: 
            raise("dataset not implemented")

        self.ds = SNAPDataset(root="/tmp/SNAP", name=ds_key)

        addSelfLoops = T.AddSelfLoops()
        self.ds.data = addSelfLoops(self.ds.data)


        y = self.ds.data.x[:,0].long()
        x = self.ds.data.x[:,1:]

        self.ds.data.y = y
        self.ds.data.x = x

        self.number_of_nodes = self.ds.data.y.size()[0]
        self.classes = len(self.ds.data.y.unique())

        l(f"Train/Test split: {train_split}/{test_split}")

        if train_split:
            train_size = train_split
        else:
            train_size = self.number_of_nodes*0.7
        
        if test_split:
            test_size = test_split
        else: 
            test_size = self.number_of_nodes*0.3
        
        train_per_class = int(train_size/self.classes)

        transform_train_test =  T.RandomNodeSplit(split="random", num_train_per_class = train_per_class,num_test=test_size, num_val=0)

        self.ds.data = transform_train_test(self.ds.data)

        d(f"Number of Training nodes: {self.ds.train_mask.sum()}")
        d(f"Number of Testing nodes: {self.ds.test_mask.sum()}")
        
        # The number of training nodes in self.ds is not the same as in train_split, the following will complete the number of nodes
        rest_train = (train_split - self.ds.data.train_mask.sum())

        if rest_train > 0:
            d(f"{rest_train} nodes need to be added to the trainig set to reach the desired {self.train} training nodes")

            indices_train = torch.nonzero((self.ds.train_mask == True), as_tuple=True)[0].numpy()
            indices_test = torch.nonzero((self.ds.test_mask == True), as_tuple=True)[0].numpy()

            choices = np.setdiff1d(range(self.ds.data.num_nodes),np.union1d(indices_train,indices_test))
            new_samples = np.random.choice(choices,rest_train.item(),replace=False)
            
            self.ds.data.train_mask[new_samples] = True
            
            d(f"Number of Training nodes: {self.ds.train_mask.sum()}")
            d(f"Number of Testing nodes: {self.ds.test_mask.sum()}")

    def get_data(self):
        return copy.deepcopy(self.ds.data)

