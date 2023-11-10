from torch.utils.data import Dataset
import datetime
import pandas as pd 
import numpy as np
from tqdm import tqdm

def transdate(times):
    return datetime.datetime.fromtimestamp(times / 1000)

class Ori_Dataset(Dataset):
    def __init__(self,path,start_longitude,start_latitude,stop_longitude,stop_latitude,longitude_lenth,latitude_lenth):
        super(Ori_Dataset,self).__init__()
        self.data = pd.read_csv(path)
        self.start_longitude = start_longitude
        self.start_latitude = start_latitude
        
        self.stop_longitude = stop_longitude
        self.stop_latitude = stop_latitude
        
        self.longitude_lenth = longitude_lenth
        self.latitude_lenth = latitude_lenth

        self.step_longitude = (stop_longitude - start_longitude) / (longitude_lenth)
        self.step_latitude = (stop_latitude - start_latitude) / (latitude_lenth)

    def split_data(self):
        start_ori = self.data['start_time']
        start_lon = self.data['start_longitude']
        start_lat = self.data['start_latitude']

        stop_ori = self.data['stop_time']
        stop_lon = self.data['stop_longitude']
        stop_lat = self.data['stop_latitude']

        start_time = [item for item in start_ori]
        stop_time = [item for item in stop_ori]

        date = start_time + stop_time
        longitude = list(start_lon) + list(stop_lon)
        latitude = list(start_lat) + list(stop_lat)

        self.start_date_series = min(date)
        self.stand_data = pd.DataFrame(dict(date=date, longitude=longitude, latitude=latitude)).sort_values(by='date')

    def count_history_series(self,one_time_step):
        # one_hour = 3600000
        # step_stop_time = 1640966400000   1640937600000
        Ori_Dataset.split_data(self)
        load_df = []
        end_date = self.start_date_series + one_time_step
        self.start_date_series = transdate(self.start_date_series)
        one_hour_load = np.zeros(self.latitude_lenth * self.longitude_lenth)

        with tqdm(total=len(self.stand_data)) as pbar :
            for item in self.stand_data.values:
                if item[0] >= end_date:
                # 重置截止日期，重置列表，添加列表
                    end_date += one_time_step
                    load_df.append(one_hour_load)

                    one_hour_load = np.zeros(self.latitude_lenth * self.longitude_lenth)

                if item[0] < end_date:
                # 确定位置
                    i = (item[1] - self.start_longitude) // self.step_longitude
                    j = (item[2] - self.start_latitude) // self.step_latitude
                    one_hour_load[int(j * self.longitude_lenth + i)] += 1
                pbar.update(1)
            load_df.append(one_hour_load)
        # date = [self.start_date_series + datetime.timedelta(hours=i) for i in range(len(load_df))]
        self.res_series = pd.DataFrame(load_df)
        self.res_series['time'] = [self.start_date_series + datetime.timedelta(hours=i) for i in range(len(load_df))]
        return self.res_series

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()

# path = '/workspace/elec_station/data/Trip2022Q1.csv'

# start_longitude = 113.702281
# start_latitude = 29.969076

# stop_longitude = 115.082572
# stop_latitude = 31.36126

# longitude_lenth = 132
# latitude_lenth = 155

# one_hour = 3600000

# mydataset = Ori_Dataset(path,
#                         start_longitude,
#                         start_latitude,
#                         stop_longitude,
#                         stop_latitude,
#                         longitude_lenth,
#                         latitude_lenth)

# all_series_dataset = mydataset.count_history_series(one_hour)
