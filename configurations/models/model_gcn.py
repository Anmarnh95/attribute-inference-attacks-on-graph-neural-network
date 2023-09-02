import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from tqdm import tqdm
from configurations.models.targetmodelinterface import TargetModelInterface
from torch_geometric.data import Data
import json as js
from logging import info as l
from logging import debug as d

class GCN(torch.nn.Module):
    def __init__(self,num_features,num_classes, hidden_layer = 16, dropout = 0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_layer)
        self.conv2 = GCNConv(hidden_layer, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class Model_GCN(TargetModelInterface):

    def __init__(self, data, model_name, dataset_name, hidden_size = 16,epochs = 200, lr = 0.01, weight_decay=5e-4, device = 'cpu', dropout = 0.5, private_parameters = None, save_path = ""):

        self.dataset_name = dataset_name
        self.data = data
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.dropout = dropout
        self.save_path = save_path

        self.wrapped_model = GCN(num_classes=len(torch.unique(self.data.y)), num_features=self.data.num_features, hidden_layer=hidden_size)

    def train_model(self):

            np.random.seed(42)
            torch.manual_seed(42)

            l(f"TRAINING NONPRIVATE {self.model_name} MODEL")
            optimizer = torch.optim.Adam(self.wrapped_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            data = self.data
            
            l(self.wrapped_model)

            self.wrapped_model.train()
            for _ in tqdm(range(self.epochs)):
                optimizer.zero_grad()
                out = self.wrapped_model(data)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

            pred_t = self.wrapped_model(data).argmax(dim=1)
            correct_t = (pred_t[data.train_mask] == data.y[data.train_mask]).sum()
            acc_t = int(correct_t) / int(data.train_mask.sum())
            l(f'Train Accuracy: {acc_t:.4f}')

            self.wrapped_model.eval()
            pred = self.wrapped_model(data).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
            l(f'Test Accuracy: {acc:.4f}')

            accur = {"Acuracy_train": acc_t ,"Acuracy_test": acc}
            json = js.dumps(accur)
            f = open(f"{self.save_path}/model_metrics_nonprivate_{self.model_name}.json","w")
            f.write(json)
            f.close()



    def prepare_model(self):
        self.train_model()

    def query_model(self, x, edge_index):
        query_data = Data(x=x.type(torch.float).to(device=self.device),edge_index=edge_index.to(device=self.device)).to(self.device)
        return torch.exp(self.wrapped_model(query_data))

    def get_acc(self, model,device,data):
        model.eval()
        data_t = self.transform_edges_form(data).to(device=device)

        pred = F.softmax(model(data_t.x,data_t.adj_t),dim=1).argmax(dim=1)
        correct = (pred[data_t.test_mask] == data_t.y[data_t.test_mask]).sum()
        acc = int(correct) / int(data_t.test_mask.sum())
        l(f'Accuracy: {acc:.4f}')

    def state_dict(self):
        return self.wrapped_model.state_dict()

    def eval(self):
        return self.wrapped_model.eval()
    
    def load_state_dict(self, loaded):
        return self.wrapped_model.load_state_dict(loaded)