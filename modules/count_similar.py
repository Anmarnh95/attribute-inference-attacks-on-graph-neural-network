from modules.sampler import Sampler
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms.remove_duplicated_edges import RemoveDuplicatedEdges

class Count_Similar():

    def __init__(self, dataset: Data, split, nodes_to_count, unknown_indx, counter):

        self.ds = dataset
        self.split = split
        self.nodes_to_count = nodes_to_count
        self.unknown_indx = unknown_indx

        indices_train = torch.flatten(torch.nonzero(dataset.train_mask)).numpy()
        self.choices = np.random.choice(indices_train,size=split,replace=False)
        
        try:
            last_saved_choices = np.load(f'choices{counter - 1}.npy')
        except FileNotFoundError:
            last_saved_choices = None
        
        if last_saved_choices is None:
            print("First Run")
        elif np.array_equal(np.sort(self.choices), np.sort(last_saved_choices)):
            print(f"Iteration {counter}: The current choices are the same as the last saved choices.")
        else:
            print(f"Iteration {counter}: The current choices are different from the last saved choices.")

        
        np.save(f'choices{counter}.npy', self.choices)

        self.subgraph = dataset.subgraph(torch.tensor(self.choices))
        removaol = RemoveDuplicatedEdges("edge_attr")
        self.subgraph = removaol(self.subgraph)

        sorted_edge_index = torch.sort(self.subgraph.edge_index, dim=0)[0]

        _, unique_indices = torch.unique(sorted_edge_index, dim=1, return_inverse=True)

        unique_edges = self.subgraph.edge_index[:, unique_indices]

        random_indices = torch.randperm(unique_edges.size(1))[:self.nodes_to_count]
        self.selected_edges = unique_edges[:, random_indices].t()
        sorted_selected_edge = torch.sort(self.selected_edges, dim=0)[0]


    def count(self):

        count_all_0 = 0
        count_all_1 = 0
        for node in self.subgraph.x:
            if node[self.unknown_indx] == 0:
                count_all_0 += 1
            else:
                count_all_1 += 1
        
        print(f"In the 500 nodes: {count_all_1} are ones and {count_all_0} are zeros.")

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
