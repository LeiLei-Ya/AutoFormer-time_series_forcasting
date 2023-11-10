import pandas as pd
from gluonts.time_feature import time_features_from_frequency_str
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from model import transformer_loader

class FeatureExtend:
    def __init__(self, df, model_config):
        self.df = df
        self.config = model_config

        self.time_columns = model_config['TIME_COLUMNS']
        self.category_column = model_config['CATEGORY_COLUMN']
        self.continuous_column = model_config['CONTINUOUS_COLUMN']
        self.station_column = model_config['STATION_COLUMN']

        self.time_type = model_config['time_type']
        self.freq = model_config['Freq']

        self.columns = [x.strip() for x in df.columns]
        self.time_features = time_features_from_frequency_str(self.config['Freq'])


    def concat_Dataframe(self, df_group, time_feature_list):
        WINDOWS = self.config['WINDOWS']
        MULTI_STEP = self.config['MULTI_STEP']
        df_all_list = list()
        with tqdm(total = len(df_group)) as pbar:
            for one_station in df_group:
                for i in range(len(self.time_features)):
                    one_station[1]['time_fea_'+str(i)] = list(time_feature_list[i])
                # train_ratio = 1
                df_one = one_station[1].sort_index(ascending=True)
                df_reframed = transformer_loader.series_to_supervised(df_one.values, WINDOWS, MULTI_STEP, df_one.columns, False)

                df_reframed.index = df_one.index
                df_reframed = df_reframed.dropna()
                df_reframed['label'] = df_reframed.apply(lambda x : [x[xx] for xx in self.furture_value_name], axis=1)

                df_reframed['is_train'] = 0
                df_reframed.loc[list(df_reframed.index)[:int(len(df_one.index))], 'is_train'] = 1
                
                for x in self.config['static_fea']:
                    df_reframed[x]=df_reframed[x+'(t+0)']
                
                df_reframed[self.station_column]=one_station[0]
                df_all_list.append(df_reframed)
                pbar.update(1)        
        df_concat_res = pd.concat(df_all_list)
        return df_concat_res

    def add_time_feature(self):
        df = self.df
        time_columns = self.time_columns
        mutil_step = self.config['MULTI_STEP']
        
        df[time_columns] = pd.to_datetime(df[self.time_columns], format=self.time_type)

        ###### maybe 这一步会有问题
        time_feature_list = [df.iloc[0:2160,:].apply(lambda x : self.time_features[i](pd.PeriodIndex([x['date']],freq = self.freq)), axis = 1) for i in range(len(self.time_features))] 
        df = df.set_index(self.time_columns)

        dict_category_encoder = {}
        for c in self.category_column:
            encoder = LabelEncoder()
            values = df[c].fillna('other')
            df[c] = encoder.fit_transform(values)
            dict_category_encoder[c] = encoder

        dict_continuous_encoder = {}
        for c in self.continuous_column:
            df[c] = df[c].fillna(0)
            continuous_scaler = MinMaxScaler(feature_range=(0, 1))
            data = continuous_scaler.fit_transform(df[c].values.reshape(-1, 1))
            df[c] = data.reshape(-1,).tolist()
            dict_continuous_encoder[c] = continuous_scaler

        self.furture_value_name = [x+'(t+%d)'%(i) for i in range(mutil_step) for x in self.config['furture_value']]
        df_all_list = []
        df_group = df.groupby(self.station_column)
        return FeatureExtend.concat_Dataframe(self,df_group, time_feature_list), self.time_features, dict_category_encoder, dict_continuous_encoder