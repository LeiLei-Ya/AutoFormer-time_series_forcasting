from transformers import AutoformerConfig, AutoformerForPrediction

class TransformerLoader():
    def __init__(self, df, batch_size):
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

