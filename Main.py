import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PreProcess as prepro

import LR
import LS
import DT
import DNN
import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import random



def find_best_model(df, tgt):
    lr, ls, dt, dnn = [], [], [], []

    for i in range(100):
        seed = random.randrange(100)
        lr.append(LR.linreg(df, tgt,seed))
        ls.append(LS.lasso(df, tgt,seed))
        dt.append(DT.dectree(df, tgt,seed))
        dnn.append(DNN.nn(df, tgt,seed))

    print(pd.DataFrame({'lr':lr}).describe())
    print(pd.DataFrame({'ls':ls}).describe())
    print(pd.DataFrame({'dt':dt}).describe())
    print(pd.DataFrame({'dnn':dnn}).describe())


tgt = 'medv'

df = prepro.Data(tgt)

y = df[tgt]

seed = 101

LR.linreg(df, tgt, seed)

LS.lasso(df,tgt, seed)

DT.dectree(df,tgt, seed)

DNN.nn(df, tgt, seed)

#find_best_model(df, tgt)


#PCA.analysis(df)







