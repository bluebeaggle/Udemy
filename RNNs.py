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






#Part 2 - Building the RNN


#Part 3 - Making the predictions and visualising the results

