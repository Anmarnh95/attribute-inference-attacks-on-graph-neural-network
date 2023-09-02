from logging import info as l
from logging import debug as d
from copy import deepcopy

import os
import torch

from torch_geometric.nn import knn_graph 
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import subgraph

import torch_geometric.transforms as T

import numpy as np

from scipy.sparse.csgraph import laplacian
from sklearn import metrics

from utils.dictToString import dictToString
from utils.mlp import *
from utils.custom_data import *

from registeration import *

from modules.sampler import Sampler
from modules.perturber import Perturber
from torch.utils.data import DataLoader



class shadow_attack_manager():
    
    def __init__(self,config,device,privacy_params_comninations, save_path = ""):
        self.config = config
        self.device = device
        self.data_target = None
        self.data_shadow = None
        self.privacy_params_comninations = privacy_params_comninations
        self.sampler = Sampler()
        self.perturber = Perturber()
        self.RAA = False
        self.save_path = save_path

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

    def prepare_SA_Datasets(self):

        # load data

        dataset_savepath_target = f"{self.save_path}/saved_dataset_{self.config.dataset_name}_SA_target.pt"
        dataset_savepath_shadow = f"{self.save_path}/saved_dataset_{self.config.dataset_name}_SA_shadow.pt"

        # Load and split dataset

        if os.path.exists(dataset_savepath_target) and os.path.exists(dataset_savepath_shadow):

            self.data_target = torch.load(dataset_savepath_target)
            self.data_shadow = torch.load(dataset_savepath_shadow)

            l("SA: SUCCESSFULY LOADED AN ALREADY SAVED TARGET/SHADOW DATASET:")
            l(self.data_target)
            l(self.data_shadow)

        else:
            dataset_savepath_normal = f"{self.save_path}/saved_dataset_{self.config.dataset_name}.pt"
            

            if os.path.exists(dataset_savepath_normal):

                data_normal = torch.load(dataset_savepath_normal)
                l("SA: SUCCESSFULY LOADED AN ALREADY SAVED NORMAL DATASET:")
                l(data_normal)

            else:
                self.dataset_loader = return_dataset_loader(dataset_name=self.config.dataset_name)(dataset_name=self.config.dataset_name,
                                                                                    train_split=self.config.split[0],
                                                                                    test_split=self.config.split[1])
                data_normal = self.dataset_loader.get_data()
                l("SA: SUCCESSFULY LOADED A NEW VERSION OF THE NORMAL DATASET:")
                l(data_normal)
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

        sub_target = subgraph(sub_nodes_target,edge_index=deepcopy(data_normal.edge_index))[0]
        sub_shadow = subgraph(sub_nodes_shadow,edge_index=deepcopy(data_normal.edge_index))[0]
        
        data_target = deepcopy(data_normal)
        data_target.edge_index = sub_target

        data_shadow = deepcopy(data_normal)
        data_shadow.edge_index = sub_shadow

        transform_isolated_nodes = T.RemoveIsolatedNodes()
        
        data_target = transform_isolated_nodes(data_target)
        data_shadow = transform_isolated_nodes(data_shadow)

        transform_train_test_target =  T.RandomNodeSplit(num_test=int((data_target.x.size()[0])*test_rate), 
                                                num_val= 0)
        
        transform_train_test_shadow =  T.RandomNodeSplit(num_test=int((data_shadow.x.size()[0])*test_rate), 
                                                num_val= 0)

        data_target = transform_train_test_target(data_target)
        data_shadow = transform_train_test_shadow(data_shadow)

        return data_target, data_shadow
           
    def run_one_SA(self, ratio, m, candidates, run_n, k, save_extention,sensetive_attr):

        if not os.path.exists(f"{self.save_path}_results/shadow"):
            os.mkdir(f"{self.save_path}_results/shadow")

        if not os.path.exists(f"{self.save_path}_results/target"):
            os.mkdir(f"{self.save_path}_results/target")
        
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
                                         private_parameters = self.privacy_params_comninations,
                                         save_path=f"{self.save_path}_results/target")
         
        shadow_model = return_target_model(self.config.shadow_model_name)(data = deepcopy(self.data_shadow),
                                         model_name=self.config.model_name, 
                                         dataset_name=self.config.dataset_name,
                                         hidden_size=self.config.hidden_size,
                                         epochs=self.config.epochs,
                                         lr=self.config.lr,
                                         weight_decay=self.config.wd,
                                         device=self.device,
                                         dropout=self.config.dropout,
                                         private_parameters = self.privacy_params_comninations,
                                         save_path=f"{self.save_path}_results/shadow")
         
        # train shadow and target models
        l("TRAINING TARGET MODEL")
        target_model.prepare_model()

        l("TRAINING SHADOW MODEL")
        shadow_model.prepare_model()

        l(f"Attack: SA, m = {m}, K = {k}, Kind = {self.RAA}, Run Number = {run_n}")

        nodes_shadow = deepcopy(self.data_shadow.x)

        """ 
        Step 5: Select candidate set with their sensitive attribute from the train_set of the shadow model 
        """
        #nodes_no_flip_shadow_idx = torch.randperm(nodes_shadow.size()[0])[:nodes_shadow.size()[0]*(1-ratio)]
        nodes_no_flip_shadow_idx = torch.randperm(nodes_shadow.size()[0])[:candidates]
        
        """ 
        Step 6: Select candidate set with their sensitive attribute from the train_set of the shadow model, 
        where the value of the sensitive attribute is flipped 
        """
        all_nodes_idx = torch.arange(0, nodes_shadow.size()[0])
        remaining_nodes_idx = torch.tensor([i for i in all_nodes_idx if i not in nodes_no_flip_shadow_idx])

        nodes_flip_shadow_idx = remaining_nodes_idx[:candidates]

        nodes_shadow[nodes_flip_shadow_idx,sensetive_attr] = 1 - nodes_shadow[nodes_flip_shadow_idx,sensetive_attr]

        d(nodes_shadow[nodes_no_flip_shadow_idx])
        d(nodes_shadow[nodes_flip_shadow_idx])

        # Prepare KNN if necesary
        if k == 0:
            attack_edge_index_no_flip = knn_graph(x=nodes_shadow[nodes_no_flip_shadow_idx],k=3)
            attack_edge_index_flip = knn_graph(x=nodes_shadow[nodes_flip_shadow_idx],k=3)
        else:
            attack_edge_index_no_flip = knn_graph(x=nodes_shadow[nodes_no_flip_shadow_idx],k=k)
            attack_edge_index_flip = knn_graph(x=nodes_shadow[nodes_flip_shadow_idx],k=k)
        
        
        """ Step 7: Query the shadow model with both candidate sets """

        posteriors_1 = shadow_model(x=nodes_shadow[nodes_no_flip_shadow_idx], edge_index=attack_edge_index_no_flip)
        posteriors_0 = shadow_model(x=nodes_shadow[nodes_flip_shadow_idx], edge_index=attack_edge_index_flip)


        """ Step 8: Label not flipped nodes as 1, label flipped nodes as 0"""

        labels_1 = [1.0]*(posteriors_1.size()[0])
        labels_0 = [0.0]*(posteriors_0.size()[0])

        shadow_posterios = torch.concat((posteriors_0,posteriors_1))
        shadow_labels = labels_0 + labels_1

        d("Shadow posterios:")
        d(shadow_posterios)
        d("Shadow labels:")
        d(shadow_labels)

        """ Step 9: Train the attack model (MLP) with data from 7 and 9 (make sure they are equal) """
        train_set = TrainDataset(x=shadow_posterios,y=shadow_labels)

        batch_size = int(candidates/4)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        mlp = MLP(in_features=shadow_posterios.size()[1], hidden_features=16,out_features=1)

        mlp = train_mlp(model=mlp,train_loader=train_loader,device=self.device,epochs=20)

        """" Step 10: Select victim set from the training set of the target dataset (their sensitive value is hidden)"""
        d("TARGET:")

        #_ , idx_target = self.sampler.sample(self.data_target,candidates=candidates)
        indices_target_train = torch.flatten(torch.nonzero(self.data_target.train_mask)).numpy()
        indices_test_train = torch.flatten(torch.nonzero(self.data_target.test_mask)).numpy()

        choices_train = np.random.choice(indices_target_train,size=50,replace=False)
        choices_test = np.random.choice(indices_test_train,size=50,replace=False)
        
        target_train_nodes = self.data_target.x[choices_train]
        target_test_nodes = self.data_target.x[choices_test]

        target_samples = torch.concat((target_train_nodes,target_test_nodes))
        ground_truth = [1.0]*50 + [0.0]*50
        d(target_samples.size())
        #pertubed_target_nodes = deepcopy(self.data_target)

        rands = torch.round(torch.rand(target_samples.size()[0],1))
        d(rands.size())
        d(target_samples[:, sensetive_attr].size())

        target_samples[:, sensetive_attr] = rands

        d("Target Info")
        d(target_samples)
        d(ground_truth)

        if k == 0:
            target_edge_index = knn_graph(x=target_samples,k=3)
        else:
            target_edge_index = knn_graph(x=target_samples,k=k)
    
        """" Step 11: Query the target model with nodes from victim set, obtain posterior"""

        posteriors_target = target_model(x=target_samples,edge_index=target_edge_index)
        

        """" Step 12: Query your attack model with the posteriors and check the membership."""
        mlp_target_out = mlp(posteriors_target).detach().flatten()

        """ Step 13: Report the accuracy (this is what we take as the correctly inferred)"""
        auc = metrics.roc_auc_score(ground_truth,mlp_target_out)
        
        file_name_string = f"AUC_SA_{save_extention}_{k}_{self.RAA}__n{run_n}m{m}s{candidates}{ratio}.pt"
        torch.save(auc,f"{self.save_path}/{file_name_string}")

        l(f"Final AUC:{auc}")