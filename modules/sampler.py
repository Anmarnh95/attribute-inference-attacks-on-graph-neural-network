import numpy as np
import copy
import torch

from logging import info as l
from logging import debug as d

class Sampler():

    def sample_from_all(self, data, num):

        # Prepare a list for nodes and a list for nodes indexes
        nodes_idx = []

        # Sample the array b
        b = np.random.choice([0,1], size=num)

        sum_1 = sum(i for i in b if i == 1)
        sum_0 = num - sum_1

        # Sample indexes from dataset
        idx_0 = np.random.choice(data.x.size()[0],size=sum_0, replace=False).tolist()


        # If b is equal to 1, sample only from training set
        train_data = data.x[data.train_mask]

        idx_training = (data.train_mask == True).nonzero(as_tuple=True)[0].tolist()
        idx_1 = []
        while(len(idx_1) < sum_1):
            ch = np.random.choice(idx_training,size=1).item()
            if (not ch in idx_0) and (not ch in idx_1):
                idx_1.append(ch)

        # for i in idx_1_train:

        #     # Check which index the node has in the whole dataset (not only in the training split). Since it is possible that two or more
        #     # # nodes could have the same features, the first match will be taken. It does not matter since we are only sampling. 
        #     idx_1.append(torch.nonzero((data.x == train_data[i]).sum(dim=1) == data.x.size(1))[0].item())

        # Append everything
        nodes_idx = idx_0 + idx_1
        nodes = data.x[nodes_idx]
        
        return (nodes, nodes_idx)

    def sample(self, data, candidates):


        #indices_train = torch.nonzero((data.train_mask), as_tuple=True)[0].numpy()
        indices_train = torch.flatten(torch.nonzero(data.train_mask)).numpy()
        d("Sampler: Indices")
        d(indices_train)
        choices = np.random.choice(indices_train,size=candidates,replace=False)

        # mask = torch.zeros(data.x.size()[0], dtype=torch.bool)
        # mask[choices] = True
        d("SAMPLER:")
        d(choices)
        d(data.x[choices])


        return (data.x[choices],choices)
        
