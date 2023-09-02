"""
Abstract class for a DatasetLoader. Implement this so your dataset can be used in the experiment. 
The data should be loaded inside of __init__ and processed. get_data should return the data object of the dataset.
"""

class DatasetLoaderInterface():

    def __init__(self, dataset_name: str, train_split:int, test_split:int):
        pass

    def get_data(self):
        pass


