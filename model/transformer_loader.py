import random

class TransformerLoader():
    def __init__(self, df, config, batch_size, time_features):
        WINDOWS = config['WINDOWS']
        MULTI_STEP = config['MULTI_STEP']
        past_value = [config['past_value']]
        
        past_value_name = [x+'(t-%d)'%(i) for i in range(WINDOWS, 0, -1) for x in [config['past_value']]]           ## 72
        past_time_name = ['time_fea_'+str(j)+'(t-%d)'%(i) for i in range(WINDOWS, 0, -1) for j in range(len(time_features))]    ##288

        furture_value_name = [x+'(t+%d)'%(i) for i in range(MULTI_STEP) for x in [config['furture_value']]]
        furture_time_name = ['time_fea_'+str(j)+'(t+%d)'%(i) for i in range(MULTI_STEP) for j in range(len(time_features))]
        static_categorical_name = [config['static_fea']]

        static_categorical_fea = df[static_categorical_name].values.reshape(len(df), len(static_categorical_name))
        past_value_fea = df[past_value_name].values.reshape(len(df), WINDOWS, len(past_value)).astype(float)
        
        past_time_fea = df[past_time_name].values.reshape(len(df), WINDOWS, len(time_features)).astype(float)
        furture_value_fea = df[furture_value_name].values.reshape(len(df), MULTI_STEP, -1).astype(float)
        furture_time_fea = df[furture_time_name].values.reshape(len(df), MULTI_STEP, -1).astype(float)

        self.pairs = [[static_categorical_fea[i, :], past_value_fea[i, :, :], past_time_fea[i, :, :], 
                       furture_value_fea[i, :, :], furture_time_fea[i, :, :]] for i in range(len(df))]
        self.batch_size = batch_size
        self.position = 0

    def load_single_pair(self):
        if self.position >= len(self.pairs):
            random.shuffle(self.pairs)
            self.position = 0
        zix = self.pairs[self.position]
        self.position += 1
        return zix

    def load(self):
        while True:
            batch_static_categorical = []
            batch_past_value = []
            batch_past_time = []
            batch_furture_value = []
            batch_furture_time = []
            
            for i in range(self.batch_size):
                zipx = self.load_single_pair()
                batch_static_categorical.append(zipx[0])
                batch_past_value.append(zipx[1])
                batch_past_time.append(zipx[2])
                batch_furture_value.append(zipx[3])
                batch_furture_time.append(zipx[4])
            yield batch_static_categorical, batch_past_value, batch_past_time, batch_furture_value, batch_furture_time

def series_to_supervised(data, n_in=1, n_out=1, columns=[], dropnan=True):
	#将时间序列转换为监督学习问题
    import pandas as pd
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    if len(columns)!=n_vars:
        columns=['var%d' % (i) for i in range(n_vars)]

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(c+'(t-%d)'%(i)) for c in columns]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        names += [(c+'(t+%d)'%(i)) for c in columns]
            
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg