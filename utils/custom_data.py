from torch.utils.data import Dataset, DataLoader
import pandas as  pd

class TrainDataset(Dataset):

    def __init__(self, x, y):
        x_pd = pd.DataFrame(x.detach().numpy())
        y_pd = pd.DataFrame(y)
        self.data = pd.concat([y_pd,x_pd],axis=1).values
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind][1:]
        y = self.data[ind][0]
        return x, y
    