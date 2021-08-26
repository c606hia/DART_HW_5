import torch
import torch.nn as nn
import pyro
import numpy as np
import pandas as pd
import math
import torch.optim as optim
from pyro.nn import PyroModule
import matplotlib.pyplot as plt

def preprocess_data(data):
    x_a = []
    x_na = []
    y_a = []
    y_na = []
    for i in range(len(data)):
        if data.iloc[i]['cont_africa'] == 1 and data.iloc[i]['rgdppc_2000'] > 0:
            x_a.append([data.iloc[i]['rugged']])
            y_a.append(math.log(data.iloc[i]['rgdppc_2000']))
        if data.iloc[i]['cont_africa'] == 0 and data.iloc[i]['rgdppc_2000'] > 0:
            x_na.append([data.iloc[i]['rugged']])
            y_na.append(math.log(data.iloc[i]['rgdppc_2000']))
    x_a = torch.from_numpy(np.array(x_a)).float()
    x_na = torch.from_numpy(np.array(x_na)).float()
    y_a = torch.from_numpy(np.array(y_a)).float()
    y_na = torch.from_numpy(np.array(y_na)).float()
    return x_a, y_a, x_na, y_na

def main():
    raw_data = pd.read_csv('./rugged_data.csv',encoding='unicode_escape')
    rug_a, gdp_a, rug_na, gdp_na = preprocess_data(raw_data)
    linear_model = PyroModule[nn.Linear](1,1)
    loss_func = nn.MSELoss(reduce='sum')
    optimizer = optim.Adam(linear_model.parameters(), lr=0.01)
    iteration = 5000
    
    for i in range(iteration):
        y_pred = linear_model(rug_a).squeeze(-1)
        loss = loss_func(y_pred,gdp_a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
            print('iteration %d, loss = %f'%(i,loss.item()))
    plt.scatter(rug_a,gdp_a,s=25,c='red',label = 'africa')
    y_pred = linear_model(rug_a).detach().numpy()
    plt.plot(rug_a,y_pred,'blue',label = 'africa_relative')
    
    for i in range(iteration):
        y_pred = linear_model(rug_na).squeeze(-1)
        loss = loss_func(y_pred,gdp_na)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
            print('iteration %d, loss = %f'%(i,loss.item()))
    plt.scatter(rug_na,gdp_na,s=15,c='green',label = 'non_africa')
    y_pred = linear_model(rug_na).detach().numpy()
    plt.plot(rug_na,y_pred,'orange',label = 'non_africa_relative')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()