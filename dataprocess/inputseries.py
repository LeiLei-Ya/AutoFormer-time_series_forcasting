from torch.utils.data import Dataset
import pandas as pd

class inputDataset(Dataset):
    def __init__(self,dataframe):
        super(inputDataset,self).__init__()
        self.df = dataframe
        self.values = dataframe.values
        self.colums = dataframe.columns.values
        self.times = dataframe['time']
        
    def restruct(self):
        load_ls = [value for item in self.values[:,:-1].T for value in item]
        colums_ls = [item for item in self.colums[:-1] for i in range(len(self.df))]
        time_ls = [str(item) for item in self.times] * (self.df.shape[1] - 1)

        print(f'colums_ls lenth: {len(colums_ls)}')
        
        return pd.DataFrame(dict(date = time_ls, series_index = colums_ls, elec_load = load_ls))

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()



