from torch_geometric.nn import knn_graph 
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.sparse.csgraph import laplacian
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import subgraph
from utils.dictToString import dictToString
import torch_geometric.transforms as T
from copy import deepcopy
import os
import torch
from registeration import *
from utils.mlp import MLP
from modules.sampler import Sampler
from modules.perturber import Perturber

from torch.utils.data import Dataset, DataLoader

from utils.custom_data import *
from utils.mlp import *

from sklearn import metrics



class shadow_attack_manager():
    
    def __init__(self,config,device,privacy_params_comninations):
        self.config = config
        self.device = device
        self.data_target = None
        self.data_shadow = None
        self.privacy_params_comninations = privacy_params_comninations
        self.sampler = Sampler()
        self.perturber = Perturber()
        self.RAA = False

    def prepare_SA_Datasets(self):

        # load data

        dataset_savepath_target = f"saved_dataset_{self.config.dataset_name}_SA_target.pt"
        dataset_savepath_shadow = f"saved_dataset_{self.config.dataset_name}_SA_shadow.pt"

        # Load and split dataset

        if os.path.exists(dataset_savepath_target) and os.path.exists(dataset_savepath_shadow):

            self.data_target = torch.load(dataset_savepath_target)
            self.data_shadow = torch.load(dataset_savepath_shadow)

            print("SA: SUCCESSFULY LOADED AN ALREADY SAVED TARGET/SHADOW DATASET:")
            print(self.data_target)
            print(self.data_shadow)

        else:
            dataset_savepath_normal = f"saved_dataset_{self.config.dataset_name}.pt"
            

            if os.path.exists(dataset_savepath_normal):

                data_normal = torch.load(dataset_savepath_normal)
                print("SA: SUCCESSFULY LOADED AN ALREADY SAVED NORMAL DATASET:")
                print(data_normal)

            else:
                self.dataset_loader = return_dataset_loader(dataset_name=self.config.dataset_name)(dataset_name=self.config.dataset_name,
                                                                                    train_split=self.config.split[0],
                                                                                    test_split=self.config.split[1])
                data_normal = self.dataset_loader.get_data()
                print("SA: SUCCESSFULY LOADED A NEW VERSION OF THE NORMAL DATASET:")
                print(data_normal)
                torch.save(data_normal,dataset_savepath_normal)
            
            self.data_target, self.data_shadow = self.split_target_shadow(data_normal=data_normal)
            torch.save(self.data_target,dataset_savepath_target)
            torch.save(self.data_shadow,dataset_savepath_shadow)


    def split_target_shadow(self, data_normal):
        
        rate = self.config.target_to_shadow_rate
        test_rate = self.config.test_rate
        total_nodes = data_normal.x.size()[0]

        sub_nodes_target = list(range(int(total_nodes*rate)))
        sub_nodes_shadow = list(range(int(total_nodes*rate),int(total_nodes)))

        sub_target = subgraph(sub_nodes_target,edge_index=data_normal.edge_index)[0]
        sub_shadow = subgraph(sub_nodes_shadow,edge_index=data_normal.edge_index)[0]
        
        data_target = deepcopy(data_normal)
        data_target.edge_index = sub_target

        data_shadow = deepcopy(data_normal)
        data_shadow.edge_index = sub_shadow

        transform_isolated_nodes = T.RemoveIsolatedNodes()
        transform_train_test =  T.RandomNodeSplit(num_test=int((total_nodes/2)*test_rate), 
                                                num_val= 0)

        data_target = transform_isolated_nodes(data_target)
        data_shadow = transform_isolated_nodes(data_shadow)

        data_target = transform_train_test(data_target)
        data_shadow = transform_train_test(data_shadow)

        return data_target, data_shadow
    
    def assign_values(self, nodes, mask):
        X = deepcopy(nodes)
        rands = torch.rand(X.size())
        if self.config.binary:
            rands = (rands >= 0.5).float()
        X[mask] = rands[mask]
        return X
    
    def create_edges(self, K, nodes, mask):
        edge_mat = None
        
        new_nodes = self.assign_first(nodes, mask)
        # Do KNN
        edge_mat = knn_graph(x=new_nodes,k=K)

        return edge_mat
    
    def assign_first(self, nodes, mask):
        #Only for replacing nands..
        X = deepcopy(nodes)
        #rands = torch.rand(X.size())
        X[mask] = -1
        return X
    
    def get_idx_of_training(self, shadow = True):
        if shadow:
            return torch.flatten(torch.nonzero(self.data_shadow.train_mask)).numpy()
        else: 
            return torch.flatten(torch.nonzero(self.data_target.train_mask)).numpy()
 

    def run_SA(self):

        if self.data_target and self.data_shadow:

            candidates_set_list = self.config.candidate_set_list
            ms = self.config.m_list
            k_list = self.config.k_list
            save_extention = dictToString(self.privacy_params_comninations)
            sensetive_attr = self.config.sensetive_attr

            for ratio in self.config.shadow_perturbation_ratio:
                for m in ms:
                    for i in candidates_set_list:
                        for run_n in self.config.run_numbers:
                            for k in k_list:
                                self.run_one_SA(ratio=ratio, 
                                                m=m,candidates=i,
                                                run_n=run_n,
                                                k=k, 
                                                save_extention=save_extention,
                                                sensetive_attr= sensetive_attr)
        else:
            raise("Target and shadow split not yet initiated")
    
    def mask_to_idx(self, mask):
        a = []
        for i,e in enumerate(mask):
            if True in e:
                a.append(i)
        return a

    def assign_known_values(self,new,original, mask):
        X = deepcopy(new).to(device = self.device)
        not_mask = mask == False
        not_mask = not_mask.to(device = self.device)
        X[not_mask] = original[not_mask]
        return X

    
    def feature_propagation(self, nodes, mask, edges, binary):

        X = self.assign_values(nodes=nodes, mask=mask).to(device=self.device)

        edge_mat = edges     
        A = to_dense_adj(edge_index=edge_mat,max_num_nodes=nodes.size(0)).squeeze().numpy()
        A = torch.from_numpy(laplacian(A, normed = True))
        A = A.to(device=self.device)
        for _ in range(self.config.fp_iter):
            out = torch.sparse.mm(A,X).to(device=self.device)
            if binary:
                out = (out >= 0.5).float()
            out = self.assign_known_values(new=out,original=nodes.to(device=self.device),mask=mask).to(device=self.device)
            X = out.to(device=self.device)

        return X
                            
    def run_one_SA(self, ratio, m, candidates, run_n, k, save_extention,sensetive_attr):
        
        # Prepare models
        target_model = return_target_model(self.config.model_name)(data = deepcopy(self.data_target),
                                         model_name=self.config.model_name, 
                                         dataset_name=self.config.dataset_name,
                                         hidden_size=self.config.hidden_size,
                                         epochs=self.config.epochs,
                                         lr=self.config.lr,
                                         weight_decay=self.config.wd,
                                         device=self.device,
                                         dropout=self.config.dropout,
                                         private_parameters = self.privacy_params_comninations)
         
        shadow_model = return_target_model(self.config.shadow_model_name)(data = deepcopy(self.data_shadow),
                                         model_name=self.config.model_name, 
                                         dataset_name=self.config.dataset_name,
                                         hidden_size=self.config.hidden_size,
                                         epochs=self.config.epochs,
                                         lr=self.config.lr,
                                         weight_decay=self.config.wd,
                                         device=self.device,
                                         dropout=self.config.dropout,
                                         private_parameters = self.privacy_params_comninations)
         
         # train shadow and target models
        print("TRAINING TARGET MODEL")
        target_model.prepare_model()

        print("TRAINING SHADOW MODEL")
        shadow_model.prepare_model()

        nodes_shadow = self.data_shadow.x
        idx_shadow = list(range(self.data_shadow.x.size()[0]))
        nodes_petrubed_shadow, mask_shadow = self.perturber.perturb(candidates = nodes_shadow,
                                                                m = m,
                                                                RAA=self.RAA,
                                                                sensetive_attr=sensetive_attr, 
                                                                perturbation_ratio= ratio)
        
        
        att_kind = self.RAA
        
        print(f"Attack: SA,m = {m} ,K = {k}, Kind = {att_kind}, Run Number = {run_n}")
        
        # Prepare KNN if necesary
        if k == 0:
            attack_edge_index = self.data_shadow.edge_index
        else:
            attack_edge_index = self.create_edges(K=k,nodes=nodes_petrubed_shadow, mask=mask_shadow)
        
        attack_nodes_with_random_vals = self.assign_values(nodes=nodes_petrubed_shadow, mask=attack_edge_index)

        shadow_out = shadow_model(x=attack_nodes_with_random_vals, edge_index=attack_edge_index)

        mask_idx = self.mask_to_idx(mask_shadow)
        not_mask_idx = np.setdiff1d(idx_shadow,mask_idx)

        posteriors_0_idx = shadow_out[mask_idx]

        posteriors_1_idx = shadow_out[not_mask_idx]

        

        labels_0 = [0.0]*(posteriors_0_idx.size()[0])

        
        labels_1 = [1.0]*(posteriors_1_idx.size()[0])


        shadow_posterios = torch.concat((posteriors_0_idx,posteriors_1_idx))
        shadow_labels = labels_0 + labels_1

        if self.config.shadow_debug:
            print("AFTER MASK:")
            print(posteriors_0_idx)
            print(posteriors_1_idx)
            print(labels_0)
            print(labels_1)
            print(shadow_posterios)
            print(shadow_labels)


        train_set = TrainDataset(x=shadow_posterios,y=shadow_labels)

        batch_size = candidates
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        mlp = MLP(in_features=shadow_posterios.size()[1], out_features=1)

        mlp = train_mlp(model=mlp,train_loader=train_loader,device=self.device,epochs=20)

        print("TARGETTTTT::")

        nodes_target, idx_target = self.sampler.sample(self.data_target,candidates=candidates)
        nodes_petrubed_target, mask_target = self.perturber.perturb(candidates = nodes_target,
                                                                m = m,
                                                                RAA=self.RAA,
                                                                sensetive_attr=sensetive_attr, 
                                                                perturbation_ratio= ratio)
        
        if self.config.shadow_debug:
            print(self.data_target.train_mask)
            print(nodes_target)
            print(idx_target)
            print(nodes_petrubed_target)
            print(mask_target)

        if k == 0:
            file_name_string = f"AUC_SA_{save_extention}_F_{att_kind}__n{run_n}m{m}s{candidates}{ratio}.pt"
             
            attack_nodes_target = deepcopy(self.data_target.x)
            for i, e in enumerate(idx_target):
                attack_nodes_target[e] = deepcopy(nodes_petrubed_target[i])
            indx_of_unknown_target = idx_target
            attack_mask_target = np.isnan(attack_nodes_target).bool()
            attack_edge_index_target = self.data_target.edge_index

        else:
            file_name_string = f"AUC_SA_{save_extention}_{k}_{att_kind}__n{run_n}m{m}s{candidates}{ratio}.pt"

            attack_nodes_target = deepcopy(nodes_petrubed_target)
            indx_of_unknown_target = [i for i in range(candidates)]
            attack_mask_target = mask_target
            attack_edge_index_target = self.create_edges(K=k,nodes=nodes_petrubed_target, mask=mask_target)
        
        if self.config.shadow_debug:
            print("attack_nodes_target")
            print(attack_nodes_target)
            print("indx_of_unknown_target")
            print(indx_of_unknown_target)
            print("attack_mask_target")
            print(attack_mask_target)
        
        
        nodes_fp_petrubed_target = self.feature_propagation(nodes=attack_nodes_target,
                                                            mask=attack_mask_target,
                                                            edges=attack_edge_index_target,
                                                            binary=self.config.binary)
        

        
        posteriors_target = target_model(x=nodes_fp_petrubed_target,edge_index=attack_edge_index_target)[indx_of_unknown_target]
        
        mlp_target_out = mlp(posteriors_target).detach().flatten()

        mask_idx_target = self.mask_to_idx(mask_target)

        ground_truth = [1.0]*candidates

        for i in mask_idx_target:
            ground_truth[i] = 0.0


        auc = metrics.roc_auc_score(ground_truth,mlp_target_out)

        torch.save(auc,file_name_string)

        if self.config.shadow_debug:
            print("FP Output")
            print(nodes_fp_petrubed_target)
            print("Posterios:")
            print(posteriors_target)
            print("mlp_target_out:")
            print(mlp_target_out)
            print("mask_idx_target")
            print(mask_idx_target)
            print("ground_truth:")
            print(ground_truth)

        print(f"Final AUC:{auc}")