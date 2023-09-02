import torch
import numpy as np
import copy
from pathlib import Path
import json as js
import sys, os

from logging import info as l
from logging import debug as d

#TODO: Needs improvements!

parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def plot_results(config):
    dataset_name = config.dataset_name
    run_numbers = config.run_numbers
    candidates_list = config.candidate_set_list
    knn_list = config.k_list
    perturbation_ratio = config.perturbation_ratio


    attack_kind = ['SAA']
    attack_method = []

    dataset_savepath = f"saved_dataset_{config.dataset_name}.pt"

    d(config.split[0])
    if config.fp_included:
        attack_method.append("FP")
    if config.ma_included:
        attack_method.append("MA")
    if config.rima_included:
        attack_method.append("RIMA")
    if config.bf_included:
        attack_method.append("BF")
    if config.ri_included:
        attack_method.append("RI")


    th = config.MA_threshold
    model_name = config.model_name

    ms = config.m_list

    binary = config.binary
    round = config.round

    map_l = torch.device('cpu')
    base_path = Path().parent
    rel_path = f"outputOfExperiment{model_name}{dataset_name}{config.split[0]}/"
    rel_results_path = f"outputOfExperiment{model_name}{dataset_name}{config.split[0]}_results/"

    if not os.path.exists(rel_results_path):
            os.mkdir(rel_results_path)

    file_path = (base_path / rel_path / dataset_savepath).resolve()
    data = torch.load(f"{file_path}",map_location=map_l)


    if not config.run_shadow_attack:

        # Load restsults tensors

        results_tesors = {}
        results_idx = {}
        results_cs = {}
        results_ts = {}
        res_cs_origin = {}
        model_matrics = {}


        for method in attack_method:
            for kind in attack_kind:
                for number in run_numbers:
                    for m in ms:
                        for ratio in perturbation_ratio:
                            for samples in candidates_list:
                                for k in knn_list:
                                    if k == 0:
                                        k_str = "F"
                                    else:
                                        k_str = f"{k}"
                                            
                                    key = f"{method}__{k_str}_{kind}__n{number}m{m}s{samples}{ratio}"

                                    file_path = (base_path / rel_path / f"results__{key}.pt").resolve()
                                    results_tesors[key] = torch.load(f"{file_path}",map_location=map_l)
                                        
                                    file_path = (base_path / rel_path / f"idx__{key}.pt").resolve()
                                    results_idx[key] = torch.load(f"{file_path}",map_location=map_l)

                                    file_path = (base_path / rel_path / f"cs__{key}.pt").resolve()
                                    results_cs[key] = torch.load(f"{file_path}",map_location=map_l)

                                    file_path = (base_path / rel_path / f"ts__{key}.pt").resolve()
                                    results_ts[key] = torch.load(f"{file_path}",map_location=map_l)

                                    #Original CS
                                    kin = kind == "RAA"
                                    file_path = (base_path / rel_path  / f"CS_Original__RAA{kin}__n{number}m{m}s{samples}{ratio}.pt").resolve()
                                    res_cs_origin[f"{kind}__n{number}m{m}s{samples}"] = torch.load(f"{file_path}",map_location=map_l)
                                        

    if not config.run_shadow_attack:

        results_t = copy.deepcopy(results_tesors)
        dataset_x = copy.deepcopy(data.x)
        res_keys = list(results_t.keys())

        distances = {}


        if binary:
            
            binary_sens_attr = len(config.sensetive_attr)
            for key in res_keys:
                indx_s = key.find("s")
                if indx_s == -1:
                    raise Exception("Something went wrong")

                samples = config.candidate_set_list[0]
                
                indx_m = key.find("m")
                if indx_m == -1:
                    raise Exception("Something went wrong")

                # Ratio
                ratio = 0 
                if "." in key:
                    index_dot = key.find(".")
                    ratio = float(key[index_dot-1:])
                else:
                    ratio = 1

                m = int(key[indx_m+1:indx_s]) + binary_sens_attr


                sub = torch.sub(results_t[key][:,config.sensetive_attr], dataset_x[results_idx[key]][:,config.sensetive_attr])

                sum_sub = torch.abs(sub).sum().item()
                        
                distances[key] = (1.0 -  (sum_sub / (samples*m*ratio))) * 100
                if distances[key] > 100:
                    d(distances[key])
                    d(key)
        else:

            loss = torch.nn.MSELoss()
            for key in res_keys:
                samples = config.candidate_set_list[0]
                distances[key] = loss(dataset_x[results_idx[key]][:,config.sensetive_attr],results_t[key][:,config.sensetive_attr]).item()

    res = {}

    if not config.run_shadow_attack:
        res["Description"] = "Percentage of correctly inferred attributes of different attackes"
        res["Perturbed Attributes"] = config.sensetive_attr
        res["Binary"] = config.binary
        if not config.binary:
            res["Min Value"] = config.min_max_dataset_values[0]
            res["Max Value"] = config.min_max_dataset_values[1]
        res["Attacks"] = {}
        for method in attack_method:
            res["Attacks"][method] = {}
            for ratio in perturbation_ratio:
                res["Attacks"][method][f"Perturbation Ratio: {ratio}"] = {}
                for m in ms:
                    res["Attacks"][method][f"Perturbation Ratio: {ratio}"][f"m: {m}"] = {}
                    for kind in attack_kind:
                        res["Attacks"][method][f"Perturbation Ratio: {ratio}"][f"m: {m}"][kind] = {}
                        for k in knn_list:
                            if k == 0:
                                k_str = "F"
                                k_new_str = "Full Graph Access"
                            else:
                                k_str = f"{k}"
                                k_new_str = f"KNN with K = {k}"
                            
                            res["Attacks"][method][f"Perturbation Ratio: {ratio}"][f"m: {m}"][kind][k_new_str] = {}
                            res["Attacks"][method][f"Perturbation Ratio: {ratio}"][f"m: {m}"][kind][k_new_str]["MEAN"] = []
                            res["Attacks"][method][f"Perturbation Ratio: {ratio}"][f"m: {m}"][kind][k_new_str]["STD"] = []

                            temp = []
                            for j in run_numbers:
                                temp.append(distances[f"{method}__{k_str}_{kind}__n{j}m{m}s{samples}{ratio}"])
                            res["Attacks"][method][f"Perturbation Ratio: {ratio}"][f"m: {m}"][kind][k_new_str]["MEAN"].append(np.mean(temp))
                            res["Attacks"][method][f"Perturbation Ratio: {ratio}"][f"m: {m}"][kind][k_new_str]["STD"].append(np.std(temp))




    if config.run_shadow_attack:

        results_auc = {}
        knn_list = config.k_list
        att_kind = False

        for number in run_numbers:
            for m in ms:
                for ratio in config.shadow_perturbation_ratio:
                    for samples in candidates_list:
                        for k in knn_list:
                            if k == 0:
                                k_str = "F"
                            else:
                                k_str = f"{k}"    
                                
                            key = f"AUC_SA__{k_str}_{att_kind}__n{number}m{m}s{samples}{ratio}"

                            file_path = (base_path / rel_path / f"{key}.pt").resolve()
                            results_auc[key] = torch.load(f"{file_path}",map_location=map_l)
        
        d(results_auc.keys())



    if config.run_shadow_attack:
        res["Attacks"] = {}
        res["Attacks"]["SA"] = {}
        res["Attacks"]["SA"]["Description"] = "AUC of shadow attack"
        for ratio in config.shadow_perturbation_ratio:
            res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"] = {}
            for m in ms:
                res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"][f"m: {m}"] = {}

                if att_kind == True:
                    attack_kind = "SAA"
                else:
                    attack_kind = "RAA"

                res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"][f"m: {m}"][attack_kind] = {}
                
                for k in knn_list:
                    if k == 0:
                        k_str = "F"
                        k_new_str = "Full Graph Access"
                    else:
                        k_str = f"{k}"
                    k_new_str = f"KNN with K = {k}"
                    
                    res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"][f"m: {m}"][attack_kind][k_new_str] = {}
                    res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"][f"m: {m}"][attack_kind][k_new_str]["MEAN"] = []
                    res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"][f"m: {m}"][attack_kind][k_new_str]["STD"] = []

                    temp = []
                    for j in run_numbers:
                        temp.append(results_auc[f"AUC_SA__{k_str}_{att_kind}__n{j}m{m}s{samples}{ratio}"])
                    res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"][f"m: {m}"][attack_kind][k_new_str]["MEAN"].append(np.mean(temp))
                    res["Attacks"]["SA"][f"Perturbation Ratio: {ratio}"][f"m: {m}"][attack_kind][k_new_str]["STD"].append(np.std(temp))


    with open(f"{rel_results_path}{model_name}{dataset_name}{config.split[0]}_MEAN_{config.sensetive_attr}.json", "w") as fp:
        js.dump(res,fp,indent = 4)