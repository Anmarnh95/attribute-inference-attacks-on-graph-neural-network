import torch
import numpy as np
import copy
from torch_geometric.nn import knn_graph 
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.sparse.csgraph import laplacian
from torch_geometric.data import Data
import torch.nn.functional as F

from utils.cs import calculate_confidence_scores

class Attacker():

    def __init__(self, pertubed_nodes=None, mask = None, indexes_of_unknown = None,target_model=None, K = 0, full_access_edge_index=None, device = 'cpu', threshhold = 0.8, binary=True,iter = 10, round = True, min_max_vals= (0,1), idx_unknown = [1]):

        self.nodes = pertubed_nodes
        self.mask = mask
        self.model = target_model
        self.device = device
        self.K = K
        self.threshhold = threshhold
        self.indexes_of_unknown = indexes_of_unknown
        self.binary = binary
        self.iter = iter
        self.rounding = round
        self.min_max_vals = min_max_vals
        self.idx_unknown = idx_unknown

        if K == 0:
            self.attack_edges = full_access_edge_index
        else:
            self.attack_edges = self.create_edges(K=K,nodes=pertubed_nodes,mask=mask)

    def run_MA(self):

        X = copy.deepcopy(self.nodes)
        mask = copy.deepcopy(self.mask)
        fixed = []
        curr_th = self.threshhold
        while(True in np.isnan(X[self.indexes_of_unknown]).bool()):
            
            x_new, cs= self.feature_propagation_with_cs(nodes=X,mask=mask)
            
            top_scores = cs.topk(cs.size(0))
            print(top_scores)
            print(fixed)
            for i in range(cs.size(0)):
                index = (top_scores[1][i]).item()
                val = (top_scores[0][i]).item()
                if ((self.indexes_of_unknown[index] not in fixed) and (val >= curr_th)):
                    fixed.append(self.indexes_of_unknown[index])
                    X[self.indexes_of_unknown[index]] = x_new[index].float()
                    mask[self.indexes_of_unknown[index]] = False
                    curr_th = self.threshhold
                    break
                else:
                    curr_th = curr_th*0.95

        cs = self.query_and_eval(x_new=X,edge_matrix=self.attack_edges,indexes_of_unknown=self.indexes_of_unknown)
        mean_cs = cs.mean()
        return (X[self.indexes_of_unknown], mean_cs)
    
    def run_RIMA(self):

        X = copy.deepcopy(self.nodes)
        mask = copy.deepcopy(self.mask)
        fixed = []
        curr_th = self.threshhold
        while(True in np.isnan(X[self.indexes_of_unknown]).bool()):
            
            x_new, cs = self.random_initialization_with_cs(nodes=X,mask=mask)
    
            top_scores = cs.topk(cs.size(0))
            print(top_scores)
            print(fixed)
            for i in range(cs.size(0)):
                index = (top_scores[1][i]).item()
                val = (top_scores[0][i]).item()
                if ((self.indexes_of_unknown[index] not in fixed) and (val >= curr_th)):
                    fixed.append(self.indexes_of_unknown[index])
                    X[self.indexes_of_unknown[index]] = x_new[index].float()
                    mask[self.indexes_of_unknown[index]] = False
                    curr_th = self.threshhold
                    break
                else:
                    curr_th = curr_th*0.95

        cs = self.query_and_eval(x_new=X,edge_matrix=self.attack_edges,indexes_of_unknown=self.indexes_of_unknown)
        mean_cs = cs.mean()
        return (X[self.indexes_of_unknown], mean_cs)


    def run_FP(self):

        out, cs = self.feature_propagation_with_cs(nodes=self.nodes,mask=self.mask)
        mean_cs = cs.mean()
        return out, mean_cs

    def run_BF(self):

        C = np.isnan(self.nodes).bool()
        indx = (C == True).nonzero()
        count = indx.size(dim=0)
        results = [self.nodes, 0]
        max_ones = 2**count
        for counter in range(max_ones):
            attack_bits = format(counter, '0{a}b'.format(a = count))
            X_new = copy.deepcopy(self.nodes)
            for i in range(count):
                X_new[indx[i][0]][indx[i][1]] = torch.tensor(int(attack_bits[i]))
            
            cs = self.query_and_eval(x_new=X_new, edge_matrix = self.attack_edges,indexes_of_unknown = self.indexes_of_unknown)
            mean = torch.mean(cs)
            if mean.item() > results[1]:
                results[0] = X_new
                results[1] = mean.item()
        return results

    def run_RI(self):
        out , cs = self.random_initialization_with_cs(nodes=self.nodes, mask=self.mask)
        cs_mean = cs.mean()
        return out, cs_mean

    def random_initialization_with_cs(self, nodes, mask):
        X = self.assign_values(nodes=nodes, mask=mask).to(device=self.device)
        cs = self.query_and_eval(x_new=X, edge_matrix=self.attack_edges,indexes_of_unknown=self.indexes_of_unknown)
        return X[self.indexes_of_unknown], cs

    def query_and_eval(self, x_new, edge_matrix,indexes_of_unknown):

        Y = self.model(x_new, edge_matrix)[indexes_of_unknown]
        cs = calculate_confidence_scores(Y=Y)
        return cs

    def feature_propagation_with_cs(self, nodes, mask):

        X = self.assign_values(nodes=nodes, mask=mask).to(device=self.device)
        
        edge_mat = self.attack_edges        
        A = to_dense_adj(edge_index=edge_mat,max_num_nodes=nodes.size(0)).squeeze().numpy()
        A = torch.from_numpy(laplacian(A, normed = True))
        A = A.to(device=self.device)
        X = X.to(device=self.device)

        for _ in range(self.iter):
            out = torch.sparse.mm(A,X).to(device=self.device)
            if self.binary:
                out = (out >= 0.5).float()
            else:
                out[:,self.idx_unknown][out[:,self.idx_unknown] < self.min_max_vals[0]] = self.min_max_vals[0]
                out[:,self.idx_unknown][out[:,self.idx_unknown] > self.min_max_vals[1]] = self.min_max_vals[1]

            if self.rounding:
                out = torch.round(out)
            out = self.assign_known_values(new=out,original=nodes.to(device=self.device),mask=mask).to(device=self.device)
            X = out.to(device=self.device)
        
        cs = self.query_and_eval(x_new=X,edge_matrix=edge_mat,indexes_of_unknown=self.indexes_of_unknown)
        return X[self.indexes_of_unknown], cs

    def create_edges(self, K, nodes, mask):
        edge_mat = None
        
        new_nodes = self.assign_first(nodes, mask)
        # Do KNN
        edge_mat = knn_graph(x=new_nodes,k=K)

        return edge_mat

    def assign_known_values(self,new,original, mask):
        X = copy.deepcopy(new).to(device = self.device)
        not_mask = mask == False
        not_mask = not_mask.to(device = self.device)
        X[not_mask] = original[not_mask]
        return X

    def assign_values(self, nodes, mask):
        X = copy.deepcopy(nodes)
        rands = torch.rand(X.size())
        if self.binary:
            rands = (rands >= 0.5).float()
        else:
            rands[:,self.idx_unknown][rands[:,self.idx_unknown] < self.min_max_vals[0]] = self.min_max_vals[0]
            rands[:,self.idx_unknown][rands[:,self.idx_unknown] > self.min_max_vals[1]] = self.min_max_vals[1]

        
        X[mask] = rands[mask]
        return X

    def assign_first(self, nodes, mask):
        #Only for replacing nands..
        X = copy.deepcopy(nodes)
        #rands = torch.rand(X.size())
        X[mask] = -1
        return X