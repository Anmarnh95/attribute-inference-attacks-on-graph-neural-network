from configurations.datasets.datasetloaderinterface import DatasetLoaderInterface
from torch_geometric.datasets import LastFMAsia
import copy
import torch
import numpy as np
import torch_geometric.transforms as T

class DatasetLoader_LastFM(DatasetLoaderInterface):

    def __init__(self, dataset_name = "LasFM", train_split = 0, test_split = 0):
        self.ds_name = dataset_name
        self.train = train_split
        self.test = test_split

        self.ds = LastFMAsia(root="/tmp/LastFMAsia")

        self.number_of_nodes = self.ds.data.y.size()[0]
        self.classes = len(self.ds.data.y.unique())

        if train_split:
            train_size = train_split
        else:
            train_size = self.number_of_nodes*0.7
        
        if test_split:
            test_size = train_split
        else: 
            test_size = self.number_of_nodes*0.3
        
        train_per_class = int(train_size/self.classes)

        transform_train_test =  T.RandomNodeSplit(split="random", num_train_per_class = train_per_class,num_test=test_size, num_val=0)

        self.ds.data = transform_train_test(self.ds.data)

        print(self.ds.data.train_mask.sum())
        print(self.ds.data.test_mask.sum())
        
        # The number of training nodes in self.ds is not the same as in train_split, the following will complete the number of nodes
        rest_train = (train_split - self.ds.data.train_mask.sum()).item()

        print(rest_train)
        assert(rest_train >= 0)
        
        indices_train = torch.nonzero((self.ds.data.train_mask == True), as_tuple=True)[0].numpy()
        indices_test = torch.nonzero((self.ds.data.test_mask == True), as_tuple=True)[0].numpy()

        choices = np.setdiff1d(range(self.ds.data.num_nodes),np.union1d(indices_train,indices_test))
        new_samples = np.random.choice(choices,rest_train,replace=False)
        
        self.ds.data.train_mask[new_samples] = True

        print(self.ds.data.train_mask.sum())
        print(self.ds.data.test_mask.sum())
        

    def get_data(self):
        return copy.deepcopy(self.ds.data)
    
        

