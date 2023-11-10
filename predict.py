from transformers import AutoformerForPrediction
from itertools import chain
import yaml, os, torch, pickle,json
import pandas as pd
from gluonts.time_feature import time_features_from_frequency_str
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from dataprocess import oridataset
from model import transformer_loader
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from dataprocess.inputseries import inputDataset

with open('config/predict_config.yaml') as file:
    pre_config = yaml.safe_load(file)
with open('config/train_config.yaml') as file:
    train_config = yaml.safe_load(file)

with open('/workspace/elec_station/data/mapping.pkl','rb') as reader:
    map_dic = pickle.load(reader)

print(pre_config)

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

time_type = pre_config['time_type']
MODEL_DIR = pre_config['MODEL_DIR']
TIME_COLUMNS = pre_config['TIME_COLUMNS']
WINDOWS = pre_config['WINDOWS']
MULTI_STEP = pre_config['MULTI_STEP']
STATION_COLUMN = pre_config['STATION_COLUMN']

past_value = pre_config['past_value']
furture_value = pre_config['furture_value']
static_fea = pre_config['static_fea']
Freq = pre_config['Freq']

extend_list = lambda x : list(chain(*x))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch = "1"
model = AutoformerForPrediction.from_pretrained(os.path.join(MODEL_DIR, epoch))

model.to(device)
model.eval()

df = autoformer_input_df
df_state = df

df.columns = [x.strip() for x in df.columns]
df[TIME_COLUMNS] = pd.to_datetime(df[TIME_COLUMNS], format=time_type)

list_res = []


time_features = time_features_from_frequency_str(Freq)
time_feature_list = [df_state.iloc[0:2160,:].apply(lambda x : time_features[i](pd.PeriodIndex([x['date']],freq = Freq)), axis = 1) for i in range(len(time_features))] 

past_value_name = [x+'(t-%d)'%(i) for i in range(WINDOWS, 0, -1) for x in past_value ]
past_time_name = ['time_fea_'+str(j)+'(t-%d)'%(i) for i in range(WINDOWS, 0, -1) for j in range(len(time_features))]

furture_value_name = [x+'(t+%d)'%(i) for i in range(MULTI_STEP) for x in furture_value]
furture_time_name = ['time_fea_'+str(j)+'(t+%d)'%(i) for i in range(MULTI_STEP) for j in range(len(time_features))]
static_categorical_name = static_fea

CATEGORY_COLUMN = ['series_index']
CONTINUOUS_COLUMN = ['elec_load']

dict_category_encoder = {}
for c in CATEGORY_COLUMN:
    encoder = LabelEncoder()
    values = df[c].fillna('other')
    df[c] = encoder.fit_transform(values)
    dict_category_encoder[c] = encoder


dict_continuous_encoder = {}
for c in CONTINUOUS_COLUMN:
    df[c] = df[c].fillna(0)
    continuous_scaler = MinMaxScaler(feature_range=(0, 1))
    data = continuous_scaler.fit_transform(df[c].values.reshape(-1, 1))
    df[c] = data.reshape(-1,).tolist()
    dict_continuous_encoder[c] = continuous_scaler

df_group = df.groupby(STATION_COLUMN)

datamap={}

with tqdm(total = len(df_group)) as pbar:
    for one_station in df_group:
        df_one = one_station[1]
        time = one_station[1].iloc[len(df_one)-1, :][TIME_COLUMNS]
        index = one_station[1].iloc[len(df_one)-1, :]['series_index']
        
        #print(one_station)
        
        for i in range(len(time_features)):
            one_station[1]['time_fea_'+str(i)] = list(time_feature_list[i])
        df_one = one_station[1]


        df_temp = pd.DataFrame({'elec_load': [0]*24, TIME_COLUMNS:[pd.to_datetime(time, format='%Y-%m-%d %H:%M')+pd.Timedelta(hours=i) for i in range(1, 25)], 'series_index':[index]*24})

        for i in range(len(time_features)):
            df_temp['time_fea_'+str(i)] = df_temp.apply(lambda x : time_features[i](pd.PeriodIndex([x[TIME_COLUMNS]], freq=Freq)), axis=1)

        # df_one = df_one.append(df_temp, ignore_index = True)
        df_one = pd.concat([df_temp,df_one],ignore_index = True)


        df_one = df_one.set_index(TIME_COLUMNS)
        data = dict_continuous_encoder['elec_load'].transform(df_one['elec_load'].values.reshape(-1, 1))
        df_one['elec_load'] = data.reshape(-1,).tolist()

        furture_value_name = [x+'(t+%d)'%(i) for i in range(MULTI_STEP) for x in furture_value]
        df_one = df_one.sort_index(ascending=True).tail(72+24)

        df_reframed = transformer_loader.series_to_supervised(df_one.values, WINDOWS, MULTI_STEP, df_one.columns, False)
        df_reframed.index = df_one.index

        df_reframed = df_reframed.dropna()
        
        df_reframed['label'] = df_reframed.apply(lambda x : [x[xx] for xx in furture_value_name], axis=1) 
        for x in static_fea:
            df_reframed[x]=df_reframed[x+'(t+0)']
        df_reframed[STATION_COLUMN]=one_station[0]

        BATCH_SIZE = 1

        test_dataloader = transformer_loader.TransformerLoader(df_reframed, train_config, batch_size=BATCH_SIZE, time_features= time_features)
        all_step = 1
        list_pred = []

        batch_static_categorical, batch_past_value, batch_past_time, batch_furture_value, batch_furture_time = next(test_dataloader.load())

        batch_static_categorical = Variable(torch.IntTensor(batch_static_categorical)).to(device)
        batch_past_time = Variable(torch.DoubleTensor(batch_past_time)).to(torch.float32).to(device)

        if len(past_value) == 1:
            batch_past_value = Variable(torch.DoubleTensor(batch_past_value).reshape(BATCH_SIZE, WINDOWS)).to(torch.float32).to(device)
        else:
            batch_past_value = Variable(torch.DoubleTensor(batch_past_value)).to(torch.float32).to(device)

        past_observed_mask = torch.ones_like(batch_past_time).to(device)
        batch_furture_time = Variable(torch.DoubleTensor(batch_furture_time)).to(torch.float32).to(device)

        outputs = model.generate(
            past_values=batch_past_value,
            past_time_features=batch_past_time,
            static_categorical_features=batch_static_categorical if len(static_fea)>0 else None ,
            future_time_features=batch_furture_time
        )
        mean_prediction = outputs.sequences.mean(dim=1).cpu().numpy().reshape(-1,)

        pred_inverse = [dict_continuous_encoder['elec_load'].inverse_transform([[x]]).tolist()[0][0] for x in mean_prediction]
        list_res.append([index, pred_inverse])

        pbar.update(1)

arr_load_predict = np.zeros([24, pre_config['longitude_lenth'] * pre_config['latitude_lenth']])
for idx,values in list_res:
    column_no = map_dic[idx]
    arr_load_predict[:,column_no] = [item * 1e10 if item > 0 else 0 for item in values]

position_arr = pd.read_csv('/workspace/elec_station/data/wuhan_position.csv',header=None)
position_arr = np.where(position_arr == 0, -np.inf, position_arr)

result = dict()
for i in range(len(arr_load_predict)):
    res = np.array(arr_load_predict[i,:]+1).reshape(pre_config['latitude_lenth'],pre_config['longitude_lenth']) * position_arr
    result[f'{i+1}'] = list(res - 1)
    
res_arr = np.zeros([pre_config['latitude_lenth'] * pre_config['longitude_lenth'], 24])
for i in range(24):
    res_arr[:,i] = [item for items in result[f'{i+1}'] for item in items]
    
res_pd = pd.DataFrame(list(res_arr))
res_pd.to_csv(pre_config['OUTPUT_DIR'])

print(f"result has been saved in: {pre_config['OUTPUT_DIR']}")

