from configurations.datasets.datasetloaderinterface import DatasetLoaderInterface
from configurations.datasets.Credit.read_credit_data import load_credit
from pathlib import Path

# Please make sure that you add credit_edges.txt to the local folder ./Credit. It has all the information to build edges.
# It is not included in this repo because it's too large for Github.

class DatasetLoader_Credit(DatasetLoaderInterface):

    def __init__(self, dataset_name: str, train_split:int, test_split:int):

        self.ds_name = dataset_name
        self.train = train_split
        self.test = test_split

        base_path = Path().parent
        rel_path = f"../configurations/datasets/Credit"
        file_path = (base_path / rel_path).resolve()
        
        # If no train or test split are given (one of them is zero), use public split
        if train_split == 0 or test_split == 0:
            self.data = load_credit(path = file_path , sens_attr="Age", predict_attr="NoDefaultNextMonth", label_number=1000)
            return
    
        self.data = load_credit(path = file_path , sens_attr="Age", predict_attr="NoDefaultNextMonth", train_to_split=(self.train,self.test))


    def get_data(self):

        return self.data
