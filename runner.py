import torch
from config import config
from experiment import Experiment


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = Experiment(config=config, device=device)
    experiment.run_experiment()