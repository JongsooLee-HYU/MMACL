from .dataset import (
    CoraCocitationDataset,
    CiteseerCocitationDataset,
    PubmedCocitationDataset
)

class DatasetLoader(object):
    def __init__(self):
        pass

    def load(self, dataset_name: str = 'cora'):
        if dataset_name == 'cora':
            return CoraCocitationDataset() 
        elif dataset_name == 'citeseer':
            return CiteseerCocitationDataset()
        elif dataset_name == 'pubmed':
            return PubmedCocitationDataset()
        else:
            assert False
