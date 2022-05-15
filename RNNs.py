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
# Reshaping     # 데이터 전처리 마지막 단계 / 차원 추가 // 
#data 예측에 필요한 자원이 있다면 추가로 넣는 단계
#numpy 배열에 차원 추가랑 동일함
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1)) # 자세한 내용은 keras 활용 // (행,열,)
print(x_train)
############################################################################################
#Part 2 - Building the RNN
#과적합을 피하기 위해 드롭아웃 정규화 진행 예정
#importing the Keras Libraries and pachages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# imtialising the RNN
regressor = Sequential()                #연속된 값을 예측하기 위해 회귀분석을 이용

# Adding the first LSTM layer and some Dropout regularisation
#과적합을 피하기 위해 Dropout 진행
#장단기 메모리 층을 추가
regressor.add(LSTM(units= 50, return_sequences=True, input_shape = (x_train.shape[1],1)))       #장단기 메모리의 수, 변환순서,IMYPUT.SHAPE)
regressor.add(Dropout(0.2))
print(regressor)
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences=True))       # input없음 / 위에서 Unit이 자동으로 인식되므로, 지정할 필요 없음
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences=True))       # Same a second layer
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units= 50))        # 더 이상 순서를 되돌아가지않기 때문에 False로 변경
regressor.add(Dropout(0.2))
print(regressor)

#Adding the ouput layer
regressor.add(Dense(units=1))

# Compiling the RNN
# 최적화기 및 손실함수를 사용하기
 #kera에서 일반적으로 {RMS_prop_optimizer}를 추천함 /회귀분석이므로 평균제곱손실함수 사용함
regressor.compile(optimizer='adam', loss = 'mean_squared_error') 

#Fitting the RNN to the Training Set
regressor.fit(x_train, y_train, epochs=100, batch_size=32)


###########################################################################################
#Part 3 - Making the predictions and visualising the results
#Getting the real stock price of 2017
#Same the training set
dataset_test = pd.read_csv('Google_stock_Price_Test.csv') #Importing csv file
real_stock_price = dataset_test.iloc[:,1:2].values             #Make Numpt array - .values (index 1)
print(type(dataset_test))
print(real_stock_price)

#getting the predicted stock price of 2017



#vicualising the results


















