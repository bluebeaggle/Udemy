#Self Organizing Map

'''
비선형으로 가득차 있는 고차원 Dataset인 고객명단과 고객정보들 사이에서
어떤 패턴을 찾아야함
'''


# Importing the libraries
import enum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:,:-1].values      #dataset의 마지막열을 제외한 세트
y = dataset.iloc[:,-1].values       #dataset의 마지막열 세트


# Feature Scaling   
# 정규화과정(Normal)을 사용할 것이며, 이는 피처가 0~1 사이값이라는 뜻
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
x = sc.fit_transform(x)


# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10,y=10,input_len = 15,sigma=1.0,learning_rate= 0.5)         #sigma = 다른 이웃간의 반지름길이
# 무게 초기화
som.random_weights_init(x)
#Training 메소드
som.train_random(data = x, num_iteration= 100)


# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']

for i,x in enumerate(x) :
    w = som.winner(x)
    plot(w[0]+0.5, 
         w[1]+0.5,
         markers[y[i]], 
         colors[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2    )

show()


# Finding the frauds
mappings = som.win_map(x)
#

































