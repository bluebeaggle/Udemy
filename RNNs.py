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

# 순환 신경망이 기억해야 할 사항을 명시하는 데이터 구조
#Creating a data structure with 60 timesteps and 1 output
#1번의 결과값은 60개의 주가를 조사하여 산출량을 예측함
#1개의 timestep은 과적합으로 이루어지며, 아무것도 배우지않음
# 20개 및 30,40개 등은 몇 가지 트렌드를 포착하기에는 부족하여 60개로 진행함
# 한달에 20개의 주가가 있기 때문에 60개는 3개월치의 주가를 의미함
# 3개월치의 주가를 읽은다음 다음날의 주가를 예측하게 됨
x_train = []        #신경망 Input
y_train = []        # Output
for i in range(60, 1258) :
    x_train.append(training_set_scaled[i-60:i, 0])  #[범위, 열의 번호]
    y_train.append(training_set_scaled[i,0])        #[범위, 열의 번호]
x_train, y_train = np.array(x_train), np.array(y_train) #Keras에 넣어야 하므로 array로 재설정
print(x_train)
print(y_train)
# Reshaping     # 데이터 전처리 마지막 단계 / 차원 추가













#Part 2 - Building the RNN


#Part 3 - Making the predictions and visualising the results

