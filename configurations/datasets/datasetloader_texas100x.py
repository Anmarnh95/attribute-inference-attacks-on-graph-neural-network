from configurations.datasets.datasetloaderinterface import DatasetLoaderInterface
import numpy as np
import pandas as pd
import time
import os
from torch_geometric.data import Data
import torch
from random import sample
from copy import deepcopy
from pathlib import Path

class CustomDataset():
    def __init__(self, x, y, x_test, y_test, num_classes, num_features):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.num_features = num_features
        
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

class DatasetLoader_Texas100X(DatasetLoaderInterface):

    def __init__(self, dataset_name = "Texas100X", train_split = 0, test_split = 0):
        self.ds_name = dataset_name
        self.train = train_split
        self.test = test_split

        base_path = Path().parent
        rel_path = f"./configurations/datasets/Texas-100X"
        self.file_path = (base_path / rel_path).resolve()
        
        features, labels, (train_idx, test_idx), candidate_idx0, candidate_idx1, candidate_idx2, num_features, num_classes, attr, data = self.create_datasets(train_split, test_split)
        self.ds = CustomDataset(features, labels, train_idx, test_idx, num_classes, num_features)
        
        self.ds.data = data
        
        self.number_of_nodes = self.ds.data.y.size()[0]
        self.classes = num_classes
        
        self.ds.data.sensitive_attr = attr
        self.ds.data.candidate_idx0 = candidate_idx0
        self.ds.data.candidate_idx1 = candidate_idx1
        self.ds.data.candidate_idx2 = candidate_idx2


    def get_data(self):
        return deepcopy(self.ds.data)

    def convert_texas(self):

        features_path = (self.file_path / "texas_100x_features.p").resolve()
        obj = pd.read_pickle(features_path)

        df = pd.DataFrame(obj, columns = range(11))
        df = df.drop(df.columns[[0]], axis=1)
        os.mkdir("./edges")
    
        mat_norm1 = df.div(np.sqrt(np.square(df).sum(axis=1)), axis=0).to_numpy()
        start = time.time()
        
        edges = 0
        for i in range((len(df) // 500)):
            print(i)
            mat_norm2 = mat_norm1[500*i:500*(i+1),:]
            sims = mat_norm1 @ mat_norm2.transpose()
    
            sims = np.argwhere(sims > 0.99999)
            sims += np.array([[0, 500*i]])
            
            edges += len(sims)
            np.savetxt("edges/foo" + str(i) + ".csv", sims, delimiter=",")
            
            if (i == (len(df) // 500)-1):
                mat_norm2 = mat_norm1[500*(i+1):,:]
                sims = mat_norm1 @ mat_norm2.transpose()
                sims = np.argwhere(sims > 0.99999)
                sims += np.array([[0, 500*(i+1)]])
                edges += len(sims)
                np.savetxt("edges/foo" + str(i+1) + ".csv", sims, delimiter=",")
        
        end = time.time()
        print(end - start)
        nodes = len(df)
        edges = (edges - nodes)/2
        print("total nodes: " + str(nodes))
        print("total edges: " + str(edges))
        print("average node degree: " + str(edges*2 / nodes))
    
    def get_features(self, sampled_nodes):

        features_desc_path = (self.file_path / "texas_100x_feature_desc.p").resolve()
        obj = pd.read_pickle(features_desc_path)

        features_path = (self.file_path / "texas_100x_features.p").resolve()
        feats = pd.read_pickle(features_path)

        feats = pd.DataFrame(feats, columns = range(11))
        feats = feats.iloc[list(sampled_nodes), :]
        feats = feats.set_axis(["THCIC_ID", 'SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', 'LENGTH_OF_STAY', 'PAT_AGE', 'PAT_STATUS', 'RACE', 'ETHNICITY', 'TOTAL_CHARGES', 'ADMITTING_DIAGNOSIS'], axis = "columns")
        obj[2][4] = 1
        obj[2][5] = 1
        obj[2][9] = 1

        del obj[0]["TOTAL_CHARGES"]
        print(feats.columns[[7]])
        feats = feats.drop(feats.columns[[7]], axis=1) # drop correlated feature
        feats = feats.drop(feats.columns[[0]], axis=1) # drop uninformative feature
        feats = self.process_features(feats, obj[0], obj[2])
        print(feats)
        
        return np.array(feats)
        
    def process_features(self, features, attribute_dict, max_attr_vals):
        """
        Returns the feature matrix after expanding the nominal features, 
        and removing the features that are not needed for model training.
        """
        features = pd.DataFrame(features)
        # for expanding categorical features
        if attribute_dict != None:
            for col in attribute_dict:
                # skip in case the sensitive feature was removed above
                if col not in features.columns:
                    print(col)
                    continue
                # to expand the non-binary categorical features
                if max_attr_vals[attribute_dict[col]] != 1:
                    features[col] *= max_attr_vals[attribute_dict[col]]
                    features[col] = pd.Categorical(features[col], categories=range(int(max_attr_vals[attribute_dict[col]])+1))
            features = pd.get_dummies(features)
        return features    
    
    def skewed_split(self):

        features_path = (self.file_path / "texas_100x_features.p").resolve()
        obj = pd.read_pickle(features_path)
        
        df = pd.DataFrame(obj, columns = range(11))
        counts = df.groupby(0).count()[1]
        np_counts = np.sort(counts.to_numpy())
        
        # skewed data that only include records from 266 hospitals with the lowest population of patients
        low_counts = counts[counts <= np_counts[265]]
        low_counts = set(low_counts.index.to_numpy().flatten())
        low_hospital_ids = df[[0]].squeeze()
        low_hospital_ids = low_hospital_ids[low_hospital_ids.isin(low_counts)].index.to_numpy()
        
        # skewed data that only include records from 7 hospitals with the highest population of patients
        high_counts = counts[counts >= np_counts[-7]]
        high_counts = set(high_counts.index.to_numpy().flatten())
        high_hospital_ids = df[[0]].squeeze()
        high_hospital_ids = high_hospital_ids[high_hospital_ids.isin(high_counts)].index.to_numpy()
        
        return set(low_hospital_ids), set(high_hospital_ids)
    
            
    def create_datasets(self, model_size = 75000, train_nodes = 50000, candidate_size = 100):
        """
        Returns the feature and label matrices, the indexes of train and test set records for the model,
        the candidate indexes for three threat models, number of features, number of classes, the
        index of the sensitive attribute (ethnicity), which is 3 in this case and the graph (data object).
        """
        edges = torch.zeros((2, 1), dtype=torch.long)
        cwd = os.getcwd()  # Get the current working directory (cwd)

        features_path = (self.file_path / "texas_100x_features.p").resolve()
        feats = pd.read_pickle(features_path)
        feat_df = pd.DataFrame(feats, columns = range(11))
        
        
        sampled_nodes = sample([i for i in range(len(feat_df.index))], model_size + candidate_size)
        
        model_idx = sampled_nodes[:model_size]
        model_train_idx = model_idx[:train_nodes]
        model_test_idx = model_idx[train_nodes:]
        
        skewed1_idx, skewed2_idx = self.skewed_split()
        
        skewed1_idx = list(skewed1_idx.difference(set(sampled_nodes)))
        skewed1_idx = sample(skewed1_idx, candidate_size)
        
        skewed2_idx = list(skewed2_idx.difference(set(sampled_nodes)))
        skewed2_idx = sample(skewed2_idx, candidate_size)
        
        # TODO: FEATURE MATRIX DOES NOT INCLUDE SAMPLES FROM SKEWED DISTRIBUTIONS, INDEXES REFER TO THE ENTIRETY OF RECORDS !!!
        sampled_nodes.extend(skewed1_idx)
        sampled_nodes.extend(skewed2_idx)
        features = self.get_features(sampled_nodes)
        num_features = features.shape[1]
        features = torch.from_numpy(features).to(torch.float)

        labels_path = (self.file_path / "texas_100x_labels.p").resolve()
        labs = pd.read_pickle(labels_path)
        
        lab_df = pd.DataFrame(labs)
        lab_df = lab_df.iloc[list(sampled_nodes), :]
        labels = lab_df.to_numpy()
        num_classes = 100
        labels = torch.from_numpy(labels)
        
        print("num nodes: " + str(len(labels)))
        
        graph = Data(x = features, edge_index = edges, y = labels.squeeze(1))

        graph.train_mask = torch.zeros(model_size + 3 * candidate_size, dtype=torch.bool)
        graph.train_mask[0:len(model_train_idx)] = True
        graph.val_mask = None
        graph.test_mask = torch.zeros(model_size + 3 * candidate_size, dtype=torch.bool)
        graph.test_mask[len(model_train_idx):len(model_train_idx) + len(model_test_idx)] = True
        graph.candidate_mask = torch.zeros(model_size + 3 * candidate_size, dtype=torch.bool)
        graph.num_classes = num_classes
        graph.num_features = num_features
        graph.num_train = train_nodes
        graph.num_test = model_size - train_nodes
            
        print("ann")

        return (features, labels, ([i for i in range(len(model_train_idx))], [i + len(model_train_idx) for i in range(len(model_test_idx))]), 
                [i + model_size for i in range(candidate_size)], [i + model_size + candidate_size for i in range(candidate_size)], 
                [i + model_size + 2 * candidate_size for i in range(candidate_size)], num_features, num_classes, 3, graph)
    
        

