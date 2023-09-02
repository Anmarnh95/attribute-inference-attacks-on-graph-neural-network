import os
from itertools import product
import torch
import copy
import random
from logging import info as l
from logging import debug as d

from registeration import *
from modules.executer import Executer
from utils.dictToString import dictToString
from modules.shadow_attack import shadow_attack_manager
from utils.plotter import plot_results

class Experiment():

    def __init__(self, config, device) :
        self.config = config
        self.device = device

    def run_experiment(self):

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        self.prepare_save_path()
        self.prepare_dataset()
        self.run_attacks_loop()
        plot_results(config=self.config)

        
    def train_and_attack(self,privacy_params_comninations):

        if not privacy_params_comninations:
            model_save_path = f"{self.save_path}/saved_model_weights_{self.config.model_name}{self.config.split[0]}.pth"
        else:
            model_save_path = f"{self.save_path}/saved_model_weights_{self.config.model_name}{self.config.split[0]}{privacy_params_comninations}.pth"

        self.model = return_target_model(self.config.model_name)(data = copy.deepcopy(self.data),
                                        model_name=self.config.model_name, 
                                        dataset_name=self.config.dataset_name,
                                        hidden_size=self.config.hidden_size,
                                        epochs=self.config.epochs,
                                        lr=self.config.lr,
                                        weight_decay=self.config.wd,
                                        device=self.device,
                                        dropout=self.config.dropout,
                                        private_parameters = privacy_params_comninations,
                                        save_path = f"{self.save_path}_results")

        # Check if model exists, if not, train a new one
        if os.path.exists(model_save_path):
            self.model.load_state_dict(torch.load(model_save_path))
            l(f"SUCCESSFULY LOADED AN ALREADY SAVED VERSION OF THE MODEL {self.config.model_name}")
            
        else:
            self.model.prepare_model()
            l(f"SUCCESSFULY TRAINED AND SAVED A NEW VERSION OF THE MODEL {self.config.model_name}")
            
            torch.save(self.model.state_dict(), model_save_path)

        self.model.eval()
        candidates_set_list = self.config.candidate_set_list
        ms = self.config.m_list
        k_list = self.config.k_list
        round_values = self.config.round

        save_extention = dictToString(privacy_params_comninations)
        l(save_extention)

        seeding_counter = self.config.random_seed
        l("===============================")
        for ratio in self.config.perturbation_ratio:
            for m in ms:
                for i in candidates_set_list:
                    for run_n in self.config.run_numbers:

                        seeding_counter = seeding_counter + 1
                        random.seed(seeding_counter)
                        np.random.seed(seeding_counter)
                        torch.manual_seed(seeding_counter)
                        
                        exc_SAA = Executer(model=self.model, 
                                           run_number = run_n, 
                                           full_dataset=copy.deepcopy(self.data), 
                                           m=m, 
                                           candidates=i,
                                           RAA=False,
                                           device=self.device,
                                           binary=self.config.binary,
                                           threshold=self.config.MA_threshold,
                                           fp_iter=self.config.fp_iter,
                                           save_extention=save_extention,
                                           sensetive_attr=self.config.sensetive_attr,
                                           round = round_values,
                                           perturbation_ratio = ratio,
                                           min_max_vals=self.config.min_max_dataset_values,
                                           idx_unknown=self.config.sensetive_attr,
                                           save_path = self.save_path,
                                           dataset_name=self.config.dataset_name)
                        l("===============================")
                        exc_SAA.cal_original_cs()
                        l("===============================")

                        for k in k_list:
                            if self.config.ma_included:
                                l("===============================")
                                exc_SAA.run_attack(attack_method="MA",K=k)

                            if self.config.fp_included:
                                l("===============================")
                                exc_SAA.run_attack(attack_method="FP",K=k)

                            if self.config.bf_included:
                                l("===============================")
                                exc_SAA.run_attack(attack_method="BF",K=k)

                            if self.config.ri_included:
                                l("===============================")
                                exc_SAA.run_attack(attack_method="RI",K=k)

                            if self.config.rima_included:
                                l("===============================")
                                exc_SAA.run_attack(attack_method="RIMA",K=k)

                        if self.config.RAA:
                            exc_RAA = Executer(model=self.model, 
                                               run_number = run_n, 
                                               full_dataset=copy.deepcopy(self.data), 
                                               m=m, 
                                               candidates=i, 
                                               RAA=True,  
                                               device=self.device, 
                                               binary=self.config.binary, 
                                               threshold=self.config.MA_threshold, 
                                               fp_iter=self.config.fp_iter, 
                                               save_extention=save_extention,
                                               sensetive_attr=self.config.sensetive_attr, 
                                               round = round_values, 
                                               perturbation_ratio = ratio,
                                               dataset_name=self.config.dataset_name)
                            l("===============================")
                            exc_RAA.cal_original_cs()
                            l("===============================")
                            for k in k_list:
                                if self.config.ma_included:
                                    l("===============================")
                                    exc_RAA.run_attack(attack_method="MA",K=k)

                                if self.config.fp_included:
                                    l("===============================")
                                    exc_RAA.run_attack(attack_method="FP",K=k)

                                if self.config.bf_included:
                                    l("===============================")
                                    exc_RAA.run_attack(attack_method="BF",K=k)

                                if self.config.ri_included:
                                    l("===============================")
                                    exc_RAA.run_attack(attack_method="RI",K=k)
                                
                                if self.config.rima_included:
                                    l("===============================")
                                    exc_RAA.run_attack(attack_method="RIMA",K=k)

    def prepare_save_path(self):

        l("==================================================")
        l(f"Preparing Save Path")

        self.save_path = f"{os.getcwd()}/outputOfExperiment{self.config.model_name}{self.config.dataset_name}{self.config.split[0]}"

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        if not os.path.exists(f"{self.save_path}_results"):
            os.mkdir(f"{self.save_path}_results")
        
        l("Output of the experiment will be saved in:")
        l(self.save_path)

    def prepare_dataset(self):

        l("==================================================")
        l(f"Preparing {self.config.dataset_name} Dataset")

        dataset_savepath = f"{self.save_path}/saved_dataset_{self.config.dataset_name}.pt"

        if os.path.exists(dataset_savepath):

            self.data = torch.load(dataset_savepath)
            l("SUCCESSFULY LOADED AN ALREADY SAVED DATASET:")
            l(self.data)

        else:
            self.dataset_loader = return_dataset_loader(dataset_name=self.config.dataset_name)(dataset_name=self.config.dataset_name,
                                                                                train_split=self.config.split[0],
                                                                                test_split=self.config.split[1])
            self.data = self.dataset_loader.get_data()
            l("SUCCESSFULY LOADED A NEW VERSION OF THE DATASET:")
            l(self.data)
            torch.save(self.data,dataset_savepath)
        
    def run_attacks_loop(self):

        l("==================================================")
        l("Will Start to run experiment")

        l("Preparing private parameters of the model")
        # Prepare private parameters and attack
        privacy_params = list(self.config.privacy_parameters.keys())
        privacy_params_comninations = list(product(*self.config.privacy_parameters.values()))
        
        
        for privacy_combi in privacy_params_comninations:
            l("==================================================")
            privacy_combi_dict = dict(zip(privacy_params,privacy_combi))

            if self.config.run_shadow_attack:
                self.sa_manager = shadow_attack_manager(config=self.config, device=self.device, privacy_params_comninations= privacy_combi_dict, save_path=self.save_path)
                self.sa_manager.prepare_SA_Datasets()
                self.sa_manager.run_SA()
                return
            else:
                self.train_and_attack(privacy_params_comninations = privacy_combi_dict)