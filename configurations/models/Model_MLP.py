import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from logging import info as l
from logging import debug as d

from tqdm import tqdm

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
        
class NeuralNetwork(torch.nn.Module):
    def __init__(self,num_features,num_classes, hidden_layer = 256, dropout = 0.5):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            nn.Linear(num_features, hidden_layer),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layer, num_classes),
        )

    def forward(self, x, training = True):
        if (training):
            x = self.flatten(x)
        else:
            x = x.x
        logits = self.linear_relu_stack(x)
        return F.log_softmax(logits, dim=1)

class Model_MLP():

    def __init__(self, data, model_name, dataset_name, save_path, hidden_size = 256, Kx = 0, Ky = 0, eps_x = np.inf, eps_y = np.inf,epochs = 1, lr = 0.01, weight_decay=1e-7, device = 'cpu', dropout = 0.5, private_parameters = None):

        self.dataset_name = dataset_name
        self.data = data
        self.model_name = model_name
        self.epochs = epochs
        #self.epochs = 1
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.dropout = dropout

        self.eps_x = eps_x
        self.eps_y = eps_y
        self.kx = Kx
        self.ky = Ky
        self.save_path = save_path
        
        
        self.train_loader = DataLoader(CustomDataset(self.data.x[self.data.train_mask], self.data.y[self.data.train_mask],
                                           self.data.x[self.data.test_mask], self.data.y[self.data.test_mask], 
                                           self.data.num_classes, self.data.num_features), 100)

        self.wrapped_model = NeuralNetwork(num_classes=self.data.num_classes, num_features=self.data.num_features, 
                                               hidden_layer=hidden_size)
        
        self.acc_test = 0
        self.acc_train = 0
        
    def __call__(self, x, edge_index):
        return self.query_model(x, edge_index)
    
    def prepare_model(self):
        if self.model_name == "GraphSAGE" or self.model_name == "GCN" or self.model_name == "MLP":
            self.train_model()

    def query_model(self, x, edge_index):
        d(edge_index)
        if self.model_name == "GraphSAGE" or self.model_name == "GCN":
            query_data = Data(x=x.type(torch.float).to(device=self.device),edge_index=edge_index.to(device=self.device)).to(self.device)
            return torch.exp(self.wrapped_model(query_data))
        else:
            query_data = Data(x=x.type(torch.float).to(device=self.device),edge_index=edge_index.to(device=self.device)).to(self.device)
            return torch.exp(self.wrapped_model(query_data, False))

    def train_model(self):

        np.random.seed(42)
        torch.manual_seed(42)

        l(f"TRAINING NONPRIVATE {self.model_name} MODEL")
        wrapped_model = self.wrapped_model.to(self.device)
        train_x = self.data.x[:self.data.num_train].to(self.device)
        train_y = self.data.y[:self.data.num_train].to(self.device)
        test_x  = self.data.x[self.data.num_train:self.data.num_train+self.data.num_test].to(self.device)
        test_y  = self.data.y[self.data.num_train:self.data.num_train+self.data.num_test].to(self.device)
        optimizer = torch.optim.Adam(self.wrapped_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        
        wrapped_model.train()
        for _ in tqdm(range(self.epochs)):
            out = wrapped_model(train_x)
            d(F.nll_loss(out, train_y))
            for x, y in self.train_loader:
                optimizer.zero_grad()
                out = wrapped_model(x)
                #print(out[data.train_mask])
                loss = F.nll_loss(out, y)
                loss.backward()
                optimizer.step()

        wrapped_model.eval()
        
        #ACCURACY ON TRAINING SET
        pred_train = wrapped_model(train_x).argmax(dim=1)
        correct_train = (pred_train == train_y).sum()
        acc_train = int(correct_train) / len(train_y)
        l(f'Train accuracy: {acc_train:.4f}')
        
        #ACCURACY ON TEST SET
        pred_test = wrapped_model(test_x).argmax(dim=1)
        correct_test = (pred_test == test_y).sum()
        acc_test = int(correct_test) / len(test_y)
        accur_test = {'Acuracy': acc_test}
        self.acc_test = accur_test

    
    def get_acc(self):
        return self.acc_test
        
    def state_dict(self):
        return self.wrapped_model.state_dict()

    def eval(self):
        return self.wrapped_model.eval()
    
    def load_state_dict(self, loaded):
        return self.wrapped_model.load_state_dict(loaded)