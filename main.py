import dataprocess.oridataset as oridataset
from dataprocess.inputseries import inputDataset
import pandas as pd
import yaml
import train
from model import add_feature,transformer_loader
import pickle

def construct_map(path,arr):
    series_idx_ls = sorted(set(arr))
    mapping = dict()
    for i in range(len(series_idx_ls)):
        mapping[i] = series_idx_ls[i]

    with open(path,'wb') as writer:
        pickle.dump(mapping,writer)
map_path = '/workspace/elec_station/data/mapping.pkl'

with open('/workspace/elec_station/config/model_config.yaml','r') as config_file:
    model_config = yaml.safe_load(config_file)

with open('/workspace/elec_station/config/dataset.yaml','r') as config_file:
    dataset_config = yaml.safe_load(config_file)

with open('/workspace/elec_station/config/train_config.yaml','r') as config_file:
    train_config = yaml.safe_load(config_file)

mydataset = oridataset.Ori_Dataset(dataset_config['ori_file_path'],
                        dataset_config['start_longitude'],
                        dataset_config['start_latitude'],
                        dataset_config['stop_longitude'],
                        dataset_config['stop_latitude'],
                        dataset_config['longitude_lenth'],
                        dataset_config['latitude_lenth'])

all_series_dataset = mydataset.count_history_series(dataset_config['one_hour'])

colum_sum = list(all_series_dataset.sum())
select_colum_index = [index for index,value in enumerate(colum_sum) if value > 100]

autoformer_predic_df = all_series_dataset.iloc[:,select_colum_index]
autoformer_predic_df['time'] = all_series_dataset.iloc[:,-1]

ml_predict_df = all_series_dataset.drop(columns=select_colum_index)

autoformer_input_df = inputDataset(autoformer_predic_df).restruct()
ml_input_df = inputDataset(ml_predict_df).restruct()

with open('/workspace/elec_station/data/autoformer_input_data.pkl','wb') as writer:
    pickle.dump(autoformer_input_df, writer)

construct_map(map_path,autoformer_input_df['series_index'])

WINDOWS = model_config['WINDOWS']
MULTI_STEP = model_config['MULTI_STEP']
time_type = model_config['time_type']

concat_dataframe, time_features, dict_category_encoder, dict_continuous_encoder = add_feature.FeatureExtend(autoformer_input_df,model_config).add_time_feature()

train_dataloader = transformer_loader.TransformerLoader(concat_dataframe[concat_dataframe['is_train']==1], config=train_config, batch_size=train_config['BATCH_SIZE'], time_features=time_features)
train_step = int(len(concat_dataframe)/train_config['BATCH_SIZE'])

train.train(train_config,dict_category_encoder,train_step,train_dataloader)