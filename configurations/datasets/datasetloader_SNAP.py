from configurations.datasets.datasetloaderinterface import DatasetLoaderInterface
from torch_geometric.datasets import SNAPDataset
import copy
import torch
import numpy as np
import torch_geometric.transforms as T

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

        print("Info about facebook:")
        print(self.ds.data)
        #removeIsolatedNodes = T.RemoveIsolatedNodes()
        #self.ds.data = removeIsolatedNodes(self.ds.data)

        addSelfLoops = T.AddSelfLoops()
        self.ds.data = addSelfLoops(self.ds.data)

        print(self.ds.data)

        y = self.ds.data.x[:,0].long()
        x = self.ds.data.x[:,1:]
        print(self.ds.data)

        self.ds.data.y = y
        self.ds.data.x = x
        print(f"Classes: {y.unique()}")
        self.number_of_nodes = self.ds.data.y.size()[0]
        self.classes = len(self.ds.data.y.unique())

        print(self.ds.data)

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
        
        # The number of training nodes in self.ds is not the same as in train_split, the following will complete the number of nodes
        rest_train = (train_split - self.ds.data.train_mask.sum())

        if rest_train > 0:
            print(rest_train)
            indices_train = torch.nonzero((self.ds.train_mask == True), as_tuple=True)[0].numpy()
            print(len(indices_train))
            indices_test = torch.nonzero((self.ds.test_mask == True), as_tuple=True)[0].numpy()
            print(len(indices_test))

            choices = np.setdiff1d(range(self.ds.data.num_nodes),np.union1d(indices_train,indices_test))
            print(f"Number of choices: {choices}")
            new_samples = np.random.choice(choices,rest_train.item(),replace=False)
            
            self.ds.data.train_mask[new_samples] = True

        print(self.ds.data)
        print(self.ds.data.train_mask.sum())
        print(self.ds.data.test_mask.sum())

    def get_data(self):
        return copy.deepcopy(self.ds.data)

