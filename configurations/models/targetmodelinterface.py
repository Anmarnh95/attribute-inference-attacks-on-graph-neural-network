import torch
from torch_geometric.data import Data


class TargetModelInterface:

    def __init__(self, data: Data,model_name: str, dataset_name: str, hidden_size: int,epochs: int, lr: float, weight_decay:int, device: str, dropout: float, private_parameters: dict, save_path: str):
        pass

    def __call__(self, x, edge_index):
        return self.query_model(x, edge_index)
    
    def prepare_model(self):
        pass

    def query_model(self, x: torch.tensor, edge_index:  torch.tensor):
        pass
    
    def state_dict(self):
        pass
    
    def eval(self):
        pass

    def load_state_dict(self, loaded):
        pass