# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import datetime

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt
import time
import math

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

from utility import generate_sequences  ,LeadsPredictor ,compile_leads ,make_predictions


import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template, url_for
#from model import encode
import pickle
import operator

torch.manual_seed(1)

leads=pd.read_csv('./leads.csv',parse_dates=[0])
leads.head()


leads.head()
print('shape of leads:', leads.shape)

df_final = compile_leads(leads)


# Resampling to 6 hrs period
df_final = df_final.resample('6H').sum()

df_final.shape

print(min(df_final.index)," -> ", max(df_final.index))

leads_df_var = df_final


seq_len = 8 # 8 x 6HR = 48 Hrs
split = 0.7

scaler = StandardScaler()
df_z = pd.DataFrame(
    scaler.fit_transform(leads_df_var),
    columns=leads_df_var.columns,
    index=leads_df_var.index
)
data = generate_sequences(df_z, seq_len)
dataset_x = torch.Tensor([x[0] for x in data])
dataset_y = torch.Tensor([x[1] for x in data])
dataset = TensorDataset(dataset_x, dataset_y)

num_features = dataset[0][0].shape[1]
output_size = dataset[0][0].shape[1]
hidden_size = 128
batch_size = 1024

train_len = int(len(dataset)*split)
lens = [train_len, len(dataset)-train_len]
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
train_ds, test_ds = random_split(dataset, lens)
trainloader = DataLoader(train_ds, batch_size=batch_size, drop_last=True, num_workers=4)
testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)





# Load trained model
model_path = './LeadsPredictor2_400_8_cpu.pth'
model = LeadsPredictor(num_features, seq_len, batch_size , hidden_size, output_size,use_cuda=False)
model.load_state_dict(torch.load(model_path))
model = model.cpu().eval()



def n_step_forecast(df, n):
    # Since the model requires the batch_size we must input the last batch_size observations
    # in the history but only retrieve the prediction for the last observation as that will be 
    # an unseen forecast.
    history = df
    for i in range(n):
        data = generate_sequences(history, seq_len)
        X = np.array([x[0] for x in data])
        fc = model(torch.Tensor(X[-batch_size:]))
        next_ = fc[-1].detach().numpy()
        next_time = history.index.max() + pd.Timedelta('6H')
        history.loc[next_time] = list(next_)
    return history

#Forecasting
#steps = 8 # Number of 6 hour periods to forecast

app = Flask(__name__)


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    int_features = [x for x in request.form.values()]
     

    #final_features = [np.array(int_features)]
    steps=int(int_features[0])
    
    history = n_step_forecast(df_z, steps)



    history_ = pd.DataFrame(
        scaler.inverse_transform(history), 
        columns=history.columns,
        index=history.index
    )
        

    history = n_step_forecast(df_z, steps)
    
    history_ = pd.DataFrame(
    scaler.inverse_transform(history), 
    columns=history.columns,
    index=history.index)
    
    history_=history_.iloc[-steps:, ]
    history_=history_.clip(lower=0)
    
    
    print('history_ dataframe Size',history_.shape)
    
    print('history_ dataframe',history_)
    
    return history_.to_html() 
    #output = round(prediction[0], 2)
    #return render_template('home.html', prediction_text="Predicted leads percentage for input datapoints : {} %".format(predPrctg[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict_proba([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)
    














