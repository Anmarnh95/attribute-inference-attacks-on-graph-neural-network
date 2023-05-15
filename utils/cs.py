import torch

def calculate_confidence_scores_alternative(Y):
    # Input must be normalized!!!

    sorted, _ = torch.sort(Y)
    max = sorted[:,-1]
    rest = sorted[:,0:-1]
    sum_rest = torch.sum(rest,dim=1)
    mean_rest = torch.div(sum_rest,rest.size(1)) 
    cs = max - mean_rest

    return cs

def calculate_confidence_scores(Y):
    # Input must be normalized!!!

    sorted, _ = torch.sort(Y)
    max = sorted[:,-1]

    return max