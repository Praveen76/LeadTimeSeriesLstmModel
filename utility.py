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







# Grouping Loan Purpose, Zip Code and Campaign

def group_purpose(series):
    s = list()
    for v in series:
        v = str(v)
        if 'pur' in v.lower():
            s.append('Purchase')
        elif 'refi' in v.lower():
            s.append('Refinance')
        elif 'home' or 'equity' in v.lower():
            s.append('Home Equity')
        else:
            s.append('Other_Purpose')
    return s

def group_campaign(series):
    s = list()
    campaign = ['TV', 'Radio', 'Internet', 'Direct Mail', 'Social Media']
    for v in series:
        if v not in campaign:
            s.append('Other_Campaign')
        else:
            s.append(v)
    return s


def group_zip(series):
    s = list()
    zips = ['75', '76', '77', '78', '79']
    for v in series:
        if v not in zips:
            s.append('Other_Zip')
        else:
            s.append(v)
    return s


# This function generates sequences of 'n' 6HR periods which will be fed to the LSTM
def generate_sequences(df, tw):
    # tw = sequence length
    data = list()
    L = len(df)
    for i in range(L-tw):
        sequence = df[i:i+tw].values
        target = df[i+tw:i+tw+1].values
        data.append((sequence, target))
    return data

def compile_leads(leads_path):
    df = leads_path.dropna()
    df['ZipCode'] = [str(zp)[:2] for zp in df['ZipCode']]
    df['ZipCode'] = group_zip(df['ZipCode'])
    df['LoanPurpose'] = group_purpose(df['LoanPurpose'])
    df['Campaign'] = group_campaign(df['LeadSourceGroup'])
    df.drop('LeadSourceGroup', axis = 1, inplace=True)
    temp = df.groupby(['DateAdded','Campaign','LoanPurpose', 'ZipCode']).size().reset_index(name='counts')
    temp1 = temp.pivot_table(index="DateAdded",columns="Campaign", values="counts", fill_value=0)
    temp2 = temp.pivot_table(index="DateAdded",columns="LoanPurpose", values="counts", fill_value=0)
    temp3 = temp.pivot_table(index="DateAdded",columns="ZipCode", values="counts", fill_value=0)
    df = temp1.join(temp2,how="inner").join(temp3, how="inner")
    df.drop(['Other_Campaign', 'Other_Zip'], axis=1, inplace = True)
    df.index = pd.DatetimeIndex(df.index)
    return df










def train(model, trainloader, testloader, num_epochs):
    
    best_model = model
    best_loss = math.inf
    ts = time.time()
    losses = list()
    for epoch in range(num_epochs):
        
        train_loss, model = sub_train_(model, trainloader) # Train
        test_loss = sub_valid_(model, testloader) # Test
        losses.append(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            best_model = model
        
        if epoch % 20 == 0:
            print('Epoch: {}, train_loss: {}, test_loss: {}'.format(
                epoch, train_loss, test_loss
            ))
            
    te = time.time()
    
    # Plot Losses
    fig, ax = plt.subplots()
    ax.plot(range(num_epochs), losses)
    plt.show()
    
    mins = int((te-ts) / 60)
    secs = int((te-ts) % 60)
    
    print('Training completed in {} minutes, {} seconds.'.format(mins, secs))
    return losses, best_model



class LeadsPredictor(nn.Module):
    
    def __init__(self, num_features, seq_len,batch_size, hidden_size, output_size,use_cuda = False):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.seq_len = seq_len
        self.output_size = output_size
        self.n_layers = 1
        self.use_cuda=use_cuda

        
        self.lstm = nn.LSTM(self.num_features,
                            self.hidden_size,
                            batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.hidden_size*seq_len, output_size)
        
        # GPU support
        # torch.cuda.is_available()

        
        # Initialize hidden state
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        cell_state = torch.zeros(self.n_layers,self.batch_size, self.hidden_size)
        
        device = 'cuda' if self.use_cuda else 'cpu'

        if self.use_cuda:
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
            
        self.hidden = (hidden_state, cell_state)
    
    # Forward propagation
    def forward(self, x):
        x, h = self.lstm(x, self.hidden)
        x = self.dropout(x.contiguous().view(self.batch_size,-1))
        x = self.fc(x)
        return x
    




def make_predictions(model, df):
    predictions = [model(inp[0]) for inp in df]
    predictions = [p.detach().numpy().squeeze() for p in predictions]
    return predictions





