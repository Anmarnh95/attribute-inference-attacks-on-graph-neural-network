from modules.sampler import Sampler
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms.remove_duplicated_edges import RemoveDuplicatedEdges

class Count_Similar():

    def __init__(self, dataset: Data, split, nodes_to_count, unknown_indx):
        self.ds = dataset
        self.split = split
        self.nodes_to_count = nodes_to_count
        self.unknown_indx = unknown_indx

        indices_train = torch.flatten(torch.nonzero(dataset.train_mask)).numpy()
        self.choices = np.random.choice(indices_train,size=split[0],replace=False)
        self.subgraph = dataset.subgraph(torch.tensor(self.choices))
        removaol = RemoveDuplicatedEdges("edge_attr")
        self.subgraph = removaol(self.subgraph)

        sorted_edge_index = torch.sort(self.subgraph.edge_index, dim=0)[0]

        _, unique_indices = torch.unique(sorted_edge_index, dim=1, return_inverse=True)

        unique_edges = self.subgraph.edge_index[:, unique_indices]

        random_indices = torch.randperm(unique_edges.size(1))[:self.nodes_to_count]
        self.selected_edges = unique_edges[:, random_indices].t()

    def count(self):

        count = 0
        count_1 = 0
        count_0 = 0

        for i, edge in enumerate(self.selected_edges):
            
            if self.ds.x[edge[0]][self.unknown_indx] == self.ds.x[edge[1]][self.unknown_indx]:
                count += 1
                if self.ds.x[edge[0]][self.unknown_indx] == 0:
                    count_0 += 1
                else:
                    count_1 += 1
        
        print(f"all similar: {count}")
        print(f"1s: {count_1}")
        print(f"0s: {count_0}")
