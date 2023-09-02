import torch
import numpy as np
import copy
import torch.nn.functional as F
import time

from modules.sampler import Sampler
from modules.perturber import Perturber
from utils.cs import calculate_confidence_scores
from modules.attacker import Attacker

from logging import info as l
from logging import debug as d

class Executer():

    def __init__(self,model,full_dataset, dataset_name,device='cpu', sensetive_attr = [],perturbation_ratio = 1,run_number = 0,m = 1, candidates = 5, RAA = False, binary = True, threshold = 0.8, fp_iter = 10,save_extention="", round = True, sa_manager = None, min_max_vals=(0,1), idx_unknown= [1], save_path = ""):
        self.sampler = Sampler()
        self.perturber = Perturber()
        self.model = model
        self.sensetive_attr = sensetive_attr
        self.ds = copy.deepcopy(full_dataset)
        self.device = device
        self.samples = candidates
        self.m_per = m
        self.m = int(m * full_dataset.num_features)
        self.RAA = RAA
        self.binary = binary
        self.run_n = run_number
        self.th = threshold
        self.fp_iter = fp_iter
        self.save_extention = save_extention
        self.round = round
        self.pertubation_ratio = perturbation_ratio
        self.sa_manager = sa_manager
        self.min_max_vals = min_max_vals
        self.idx_unknown = idx_unknown
        self.save_path = save_path
        
        # Sample and Petrub
        if dataset_name == "Texas100X":
            self.idx_ds = np.random.choice(self.ds.candidate_idx2, candidates)
            self.nodes_ds = self.ds.x[self.idx_ds]
        else:
            self.nodes_ds, self.idx_ds = self.sampler.sample(self.ds,candidates)
            
        self.nodes_petrubed, self.mask = self.perturber.perturb(candidates = self.nodes_ds,m= self.m,RAA=RAA, sensetive_attr=self.sensetive_attr, perturbation_ratio= self.pertubation_ratio)

    
    def cal_original_cs(self):

        file_name_cs = f"{self.save_path}/CS_Original_{self.save_extention}_RAA{self.RAA}__n{self.run_n}m{self.m_per}s{self.samples}{self.pertubation_ratio}.pt"
        Y = self.model(self.ds.x, self.ds.edge_index)
        Y = Y[self.idx_ds]
        cs_mean = calculate_confidence_scores(Y=Y).mean()
        l(f"CS of original values in RAA {self.RAA} is: {cs_mean}")
        torch.save(cs_mean,file_name_cs)

    def run_attack(self, attack_method = "MA", K = 2):

        att_kind = "RAA" if self.RAA else "SAA"
        l(f"Attack: {attack_method},m = {self.m_per} ,K = {K}, Kind = {att_kind}, Run Number = {self.run_n}")
        
        attack_nodes = None
        indx_of_unknown = []
        attack_mask = None
        if K == 0:
            file_name_string = f"{attack_method}_{self.save_extention}_F_{att_kind}__n{self.run_n}m{self.m_per}s{self.samples}{self.pertubation_ratio}.pt"
            
            attack_nodes = copy.deepcopy(self.ds.x)
            for i, e in enumerate(self.idx_ds):
                attack_nodes[e] = copy.deepcopy(self.nodes_petrubed[i])
            indx_of_unknown = self.idx_ds
            attack_mask = np.isnan(attack_nodes).bool()
        else:
            file_name_string = f"{attack_method}_{self.save_extention}_{K}_{att_kind}__n{self.run_n}m{self.m_per}s{self.samples}{self.pertubation_ratio}.pt"

            attack_nodes = copy.deepcopy(self.nodes_petrubed)
            indx_of_unknown = list(range(self.samples))
            attack_mask = self.mask

        d("Attack Nodes")
        d(attack_nodes)
        d("Attack Mask:")
        d(attack_mask)
        d("indx_of_unknown:")
        d(indx_of_unknown)

        l(f"Attack {method} will start")

        attacker = Attacker(pertubed_nodes=attack_nodes,
                            mask=attack_mask,
                            indexes_of_unknown=indx_of_unknown,
                            target_model=self.model,
                            K=K,
                            full_access_edge_index=self.ds.edge_index,
                            device=self.device,
                            threshhold=self.th, 
                            binary=self.binary,
                            iter = self.fp_iter,
                            round=self.round,
                            min_max_vals = self.min_max_vals,
                            idx_unknown=self.idx_unknown)
        
        l(f"Attack  {attack_method} in progress")
        start = time.time()
        if attack_method == "MA":
            out, mean_cs = attacker.run_MA()
        elif attack_method == "FP":
            out, mean_cs = attacker.run_FP()
        elif attack_method == "BF":
            out, mean_cs = attacker.run_BF()
        elif attack_method == "RI":
            out, mean_cs = attacker.run_RI()
        elif attack_method == "RIMA":
            out, mean_cs = attacker.run_RIMA()
        else:
            raise Exception("Attack method not given or wrong.")
        end = time.time()

        
        timestamp = float(format(end - start))

        l(f"Attack {attack_method} is over")
        d(f"Output of {attack_method}:{out}")
        l(f"Mean Confidence of {attack_method}: {mean_cs}")
        d(f"Indexes in Dataset of {attack_method}: {self.idx_ds}")
        
        # Save results
        torch.save(out,f"{self.save_path}/results__{file_name_string}")
        torch.save(mean_cs,f"{self.save_path}/cs__{file_name_string}")
        torch.save(self.idx_ds,f"{self.save_path}/idx__{file_name_string}")
        ts = torch.tensor([timestamp])
        l(f"Timestamp of {attack_method}: {ts}")
        torch.save(ts, f"{self.save_path}/ts__{file_name_string}")
