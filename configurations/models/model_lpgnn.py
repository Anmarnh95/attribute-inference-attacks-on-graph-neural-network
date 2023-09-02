import torch
import torch.nn.functional as F
import numpy as np
from configurations.models.targetmodelinterface import TargetModelInterface
from torch_geometric.data import Data
import json as js
import copy
from typing import Union
from torch_geometric.transforms import ToSparseTensor
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from logging import info as l
from logging import debug as d

# WARNING: please make sure that you have LPGNN installed or cloned. Uncomment the following if you cloned it and
# make sure it's in the right directory!

# from LPGNN.LPGNN.models import NodeClassifier
# from LPGNN.LPGNN.trainer import Trainer
# from LPGNN.LPGNN.transforms import FeaturePerturbation, FeatureTransform, LabelPerturbation

class Model_LPGNN(TargetModelInterface):

    def __init__(self, data, model_name, dataset_name, hidden_size = 16, epochs = 200, lr = 0.01, weight_decay=5e-4, device = 'cpu', dropout = 0.5, private_parameters = {"Kx":0, "Ky":0, "eps_x": np.inf, "eps_y": np.inf}, save_path = ""):
        self.dataset_name = dataset_name
        self.data = data
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.dropout = dropout
        self.save_path = save_path

        self.eps_x = private_parameters["eps_x"]
        self.eps_y = private_parameters["eps_y"]
        self.kx = private_parameters["Kx"]
        self.ky = private_parameters["Ky"]

        self.wrapped_model = NodeClassifier(input_dim=self.data.num_features, num_classes=len(torch.unique(self.data.y)),x_steps=self.kx,y_steps=self.ky, dropout = self.dropout, hidden_dim = hidden_size)


    def train_model(self):

        np.random.seed(42)
        torch.manual_seed(42)

        l(f"TRAINING LPGNN MODEL: ex {self.eps_x} ey {self.eps_y}")
        trans_data = self.transform_edges_form(copy.deepcopy(self.data))
        data = trans_data.clone().to(device=self.device)
        data.name = self.data
        data.num_classes = int(data.y.max().item()) + 1
        l(data)

        data = FeatureTransform()(data=data)
        data = FeaturePerturbation(x_eps=self.eps_x)(data=data)
        data = LabelPerturbation(y_eps=self.eps_y)(data=data)
        l(f"LPGNN Number of features: {data.num_node_features}")
        l(f"LPGNN Number of classes: {data.num_classes}")
        trainer = Trainer(device=self.device)
        d(data)
        ret = trainer.fit(self.wrapped_model,data)

        d(ret)
    
        json = js.dumps(ret)
        f = open(f"{self.save_path}_results/model_metrics_ex{self.eps_x}ey{self.eps_y}.json","w")
        f.write(json)
        f.close()
        self.wrapped_model.forward_correction = False
    
    def prepare_model(self):
            self.train_model()

    def query_model(self, x, edge_index):
        query_data = Data(x=x.type(torch.float).to(device=self.device),edge_index=edge_index.to(device=self.device)).to(self.device)
        t_data = self.transform_edges_form(query_data).to(device=self.device)
        out = F.softmax(self.wrapped_model.gnn(t_data.x,t_data.adj_t), dim=1)
        return out

    def get_acc(self, model,device,data):
        model.eval()
        data_t = self.transform_edges_form(data).to(device=device)

        pred = F.softmax(model(data_t.x,data_t.adj_t),dim=1).argmax(dim=1)
        correct = (pred[data_t.test_mask] == data_t.y[data_t.test_mask]).sum()
        acc = int(correct) / int(data_t.test_mask.sum())
        l(f'Accuracy: {acc:.4f}')

    def transform_edges_form(self, old_data: Union[Data, HeteroData]):
        copied_data = copy.deepcopy(old_data)
        return ToSparseTensor()(copied_data)

    def state_dict(self):
        return self.wrapped_model.state_dict()

    def eval(self):
        return self.wrapped_model.eval()
    
    def load_state_dict(self, loaded):
        return self.wrapped_model.load_state_dict(loaded)