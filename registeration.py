import numpy as np

#__________________________________________________________________________
# Available Model Types

def return_target_model(model_name = "GCN"):
    if model_name == "GCN":
        from configurations.models.model_gcn import Model_GCN
        return Model_GCN
    elif model_name == "SAGE":
        from configurations.models.model_sage import Model_SAGE
        return Model_SAGE
    elif model_name == "LPGNN":
        # WARNING: Please read the comments in configurations.models.model_lpgnn before choosing LPGNN!
        from configurations.models.model_lpgnn import Model_LPGNN
        return Model_LPGNN
    else:
        raise("Unknown Target Model: The model chosen is not found. Either the name is wrong or the model is not implemented")

#__________________________________________________________________________
# Available Datasets

def return_dataset_loader(dataset_name = "Cora"):
    if dataset_name == "Cora" or dataset_name == "Pubmed" or dataset_name == "CiteSeer":
        from configurations.datasets.datasetLoader_planetoid import DatasetLoader_Planetoid
        return DatasetLoader_Planetoid
    elif dataset_name == "Credit":
        from configurations.datasets.datasetloader_credit import DatasetLoader_Credit
        return DatasetLoader_Credit
    elif dataset_name == "Test":
        from configurations.datasets.datasetloader_testdataset import DatasetLoader_TestDataset
        return DatasetLoader_TestDataset
    elif dataset_name == "LastFM":
        from configurations.datasets.datasetloader_lastfm import DatasetLoader_LastFM
        return DatasetLoader_LastFM
    elif dataset_name == "Facebook":
        from configurations.datasets.datasetloader_SNAP import DatasetLoader_SNAP
        return DatasetLoader_SNAP
    else:
        raise("Unknown Dataset: The dataset chosen is not found. Either the name is wrong or the model is not implemented")
