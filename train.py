from torch.autograd import Variable
from transformers import AutoformerConfig, AutoformerForPrediction
from tqdm import tqdm
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os

def train(config, dict_category_encoder, train_step, train_dataloader):
    
    EPOCH = config['EPOCH']
    LR = config['LR']
    EPSILON = config['EPSILON']
    WARMUP_STEPS = config['WARMUP_STEPS']
    MODEL_DIR = config['MODEL_DIR']
    static_fea = config['static_fea']
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    plot_losses = []

    configuration = AutoformerConfig(
    prediction_length=config['MULTI_STEP'], 
    context_length=config['WINDOWS']-max(config['lags_sequence']),
    input_size = 1,
    lags_sequence = config['lags_sequence'],
    num_time_features = 4,
    num_static_categorical_features = 1,
    cardinality = [len(dict_category_encoder[x].classes_) for x in ['series_index']],
    label_length=config['WINDOWS']-max(config['lags_sequence'])-1
    )
    model = AutoformerForPrediction(configuration)

    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr = LR, eps = EPSILON)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP_STEPS, 
            num_training_steps = train_step*EPOCH)

    for epoch in range(config['EPOCH']):
        print_loss_total = 0
        with tqdm(total=train_step, desc=f'Train Epoch {epoch}/{EPOCH}', postfix=dict) as pbar:
            for i in range(train_step):
                optimizer.zero_grad()
                
                batch_static_categorical, batch_past_value, batch_past_time, batch_furture_value, batch_furture_time = next(train_dataloader.load())
                batch_static_categorical = Variable(torch.IntTensor(batch_static_categorical)).to(device)
                
                # if len(past_value) == 1:
                batch_past_value = Variable(torch.DoubleTensor(batch_past_value).reshape(config['BATCH_SIZE'], config['WINDOWS'])).to(torch.float32).to(device)
                batch_furture_value = Variable(torch.DoubleTensor(batch_furture_value).reshape(config['BATCH_SIZE'], config['MULTI_STEP'])).to(torch.float32).to(device)

                # else:
                #     batch_past_value = Variable(torch.DoubleTensor(batch_past_value)).to(torch.float32).to(device)
                #     batch_furture_value = Variable(torch.DoubleTensor(batch_furture_value)).to(torch.float32).to(device)
                batch_furture_time = Variable(torch.DoubleTensor(batch_furture_time)).to(torch.float32).to(device)
                batch_past_time = Variable(torch.DoubleTensor(batch_past_time)).to(torch.float32).to(device)
                past_observed_mask = torch.ones_like(batch_past_value).to(device)
                
                outputs = model(
                    past_values=batch_past_value,
                    past_time_features=batch_past_time,
                    past_observed_mask = past_observed_mask,
                    static_categorical_features=batch_static_categorical if len(static_fea)>0 else None ,
                    future_values=batch_furture_value,
                    future_time_features=batch_furture_time,
                    output_hidden_states=True,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss = loss.item()
                print_loss_total += loss
                pbar.set_postfix(**{'Train Loss' : print_loss_total/(i+1),
                                    'every loss': loss })
                pbar.update(1)
        plot_losses.append(float(print_loss_total/(i+1)))
        
        output_dir = os.path.join(MODEL_DIR, str(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)