'''
영화를 좋아할지 아닐지에 대한 예측시스템과 평점 예측 시스템
볼츠만 머신 - 영화 추천시스템
자동 인코더 - 평점 예측

두개의 추천시스템을 만들지만, 동일한 데이터 사전 처리과정을 진행함
'''
'''Download Dataset'''

'''전처리 과정 - 
데이터 불러오기, 훈련세트&데스트세트 준비,사용자수&영화수 Counting,
사용자를 행으로 영화를 열로 변환, 데이터를 torch.Tensor로 변환'''

"""##Importing the libraries"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel            # 병렬계산
import torch.optim as optim         # 옵티마이저
import torch.utils.data
from torch.autograd import Variable # 확률적 범위 와 하강

# Importing the dataset
movies = pd.read_csv(r'C:\Users\Daeheon\Desktop\Daeheon\PRO\All of  DeepLearning\P16-Boltzmann-Machines\Boltzmann_Machines\ml-1m\movies.dat', sep = '::', header = None, engine= 'python', encoding= 'latin-1')
users = pd.read_csv(r'C:\Users\Daeheon\Desktop\Daeheon\PRO\All of  DeepLearning\P16-Boltzmann-Machines\Boltzmann_Machines\ml-1m\users.dat', sep = '::', header = None, engine= 'python', encoding= 'latin-1')
ratings = pd.read_csv(r'C:\Users\Daeheon\Desktop\Daeheon\PRO\All of  DeepLearning\P16-Boltzmann-Machines\Boltzmann_Machines\ml-1m\ratings.dat', sep = '::', header = None, engine= 'python', encoding= 'latin-1')

# Preparing the Training set and the test set

























































