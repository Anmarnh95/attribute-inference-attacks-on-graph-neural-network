from configurations.datasets.datasetloaderinterface import DatasetLoaderInterface
from torch_geometric.data import Data
import torch
from torch_geometric.nn import knn_graph
import numpy as np


def load_testdataset(train = 5, test = 2):

    x = torch.tensor([[1, 0, 1, 0], 
                    [0, 1, 0, 1],
                    [1, 0, 1, 0], 
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0], 
                    [0, 1, 0, 1],
                    [1, 0, 1, 0], 
                    [0, 1, 0, 1],], dtype=torch.float)
    labels = torch.tensor([1,0,1,0,1,0,1,0,1,0],dtype=torch.long)

    edge_index = None

    edge_index = knn_graph(x=x,k=3)

    train_idx = np.random.choice(range(10),train,replace=False)
    test_idx = np.random.choice(np.setdiff1d(range(10), train_idx),train,replace=False)

    train_mask = torch.zeros(10, dtype=torch.bool)
    test_mask = torch.zeros(10, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    return Data(x=x, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask, val_mask=None)


class DatasetLoader_TestDataset(DatasetLoaderInterface):

    def __init__(self, dataset_name: str, train_split:int, test_split:int):

        self.ds_name = dataset_name
        self.train = train_split
        self.test = test_split

        
        # If no train or test split are given (one of them is zero), use public split
        if train_split == 0 or test_split == 0:
            self.data = load_testdataset()
            return
    
        self.data = load_testdataset(train = train_split, test = test_split)
        print("Dataset:")
        print(self.data)
        print("x:")
        print(self.data.x)
        print("y:")
        print(self.data.y)
        print("edges:")
        print(self.data.edge_index)
        print("train mask")
        print(self.data.train_mask)
        print("test mask")
        print(self.data.test_mask)

    def get_data(self):
        return self.data
