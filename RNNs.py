# Predict the stock price of Google


#Part1 - Data Preprocessing

#Importing the Libraries
import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
# Only Numnpy array can be the input of neural networks in keras
dataset_train = pd.read_csv('Google_stock_Price_Train.csv') #Importing csv file
training_set = dataset_train.iloc[:,1:2].values             #Make Numpt array - .values (index 1)
print(type(training_set))
print(training_set)

#Feature Scaling 
#Recommand Normalisation when Sigmoid Activation Function

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range =  (0,1))       #증감 주기는 between 0 and 1
training_set_scaled = sc.fit_transform(training_set)    #정규화

print(type(training_set_scaled))
print(training_set_scaled)

#Creating a data structure with 60 timesteps and 1 output
# 순환 신경망이 기억해야 할 사항을 명시하는 데이터 구조






#Part 2 - Building the RNN


#Part 3 - Making the predictions and visualising the results

