import torch
from config import config
from experiment import Experiment
import logging


if __name__ == "__main__":
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=config.log_level, format='%(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = Experiment(config=config, device=device)
    experiment.run_experiment()

    
