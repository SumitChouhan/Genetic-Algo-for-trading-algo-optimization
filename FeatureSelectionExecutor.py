import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats
import os

#Modules for Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
from FeatureSelection import FeatureSelection
import re

fileName = 'df2_combined.csv'
df = pd.read_csv(fileName)
df.index = pd.to_datetime(df.Timestamp)
df = df[df.index.year <= 2016]


features = ['Checkpoint', 'PSAR', 'SMBA', 'OLS', 'Candle_height', 'OLS_Val_min', 'OLS_Val_max', 'OLS_Val_mean', 'OLS_Val', 'PSARVal_Momentum_5_min', 'PSARVal_Momentum_5_max', 'PSARVal_Momentum_5_mean', 'PSARVal_Momentum_5', 'PSARVal_Momentum_10_min', 'PSARVal_Momentum_10_max', 'PSARVal_Momentum_10_mean', 'PSARVal_Momentum_10', 'PSARVal_Momentum_15_min', 'PSARVal_Momentum_15_max', 'PSARVal_Momentum_15_mean', 'PSARVal_Momentum_15', 'PSARVal_Momentum_20_min', 'PSARVal_Momentum_20_max', 'PSARVal_Momentum_20_mean', 'PSARVal_Momentum_20', 'PSARVal_Momentum_25_min', 'PSARVal_Momentum_25_max', 'PSARVal_Momentum_25_mean', 'PSARVal_Momentum_25', 'PSAR_ValDiff_min', 'PSAR_ValDiff_max', 'PSAR_ValDiff_mean', 'PSAR_ValDiff', 'PSAR_ValDiff_Momentum_5_min', 'PSAR_ValDiff_Momentum_5_max', 'PSAR_ValDiff_Momentum_5_mean', 'PSAR_ValDiff_Momentum_5', 'PSAR_ValDiff_Momentum_10_min', 'PSAR_ValDiff_Momentum_10_max', 'PSAR_ValDiff_Momentum_10_mean', 'PSAR_ValDiff_Momentum_10', 'PSAR_ValDiff_Momentum_15_min', 'PSAR_ValDiff_Momentum_15_max', 'PSAR_ValDiff_Momentum_15_mean', 'PSAR_ValDiff_Momentum_15', 'PSAR_ValDiff_Momentum_20_min', 'PSAR_ValDiff_Momentum_20_max', 'PSAR_ValDiff_Momentum_20_mean', 'PSAR_ValDiff_Momentum_20', 'PSAR_ValDiff_Momentum_25_min', 'PSAR_ValDiff_Momentum_25_max', 'PSAR_ValDiff_Momentum_25_mean', 'PSAR_ValDiff_Momentum_25', 'SMBA_Diff_min', 'SMBA_Diff_max', 'SMBA_Diff_mean', 'SMBA_Diff', 'SMBA_Diff_Momentum_5_min', 'SMBA_Diff_Momentum_5_max', 'SMBA_Diff_Momentum_5_mean', 'SMBA_Diff_Momentum_5', 'SMBA_Diff_Momentum_10_min', 'SMBA_Diff_Momentum_10_max', 'SMBA_Diff_Momentum_10_mean', 'SMBA_Diff_Momentum_10', 'SMBA_Diff_Momentum_15_min', 'SMBA_Diff_Momentum_15_max', 'SMBA_Diff_Momentum_15_mean', 'SMBA_Diff_Momentum_15', 'SMBA_Diff_Momentum_20_min', 'SMBA_Diff_Momentum_20_max', 'SMBA_Diff_Momentum_20_mean', 'SMBA_Diff_Momentum_20', 'SMBA_Diff_Momentum_25_min', 'SMBA_Diff_Momentum_25_max', 'SMBA_Diff_Momentum_25_mean', 'SMBA_Diff_Momentum_25', 'std_pastData_5_min', 'std_pastData_5_max', 'std_pastData_5_mean', 'std_pastData_5', 'std_pastData_10_min', 'std_pastData_10_max', 'std_pastData_10_mean', 'std_pastData_10', 'std_pastData_15_min', 'std_pastData_15_max', 'std_pastData_15_mean', 'std_pastData_15', 'std_pastData_20_min', 'std_pastData_20_max', 'std_pastData_20_mean', 'std_pastData_20', 'std_pastData_25_min', 'std_pastData_25_max', 'std_pastData_25_mean', 'std_pastData_25', 'std_pastData_30_min', 'std_pastData_30_max', 'std_pastData_30_mean', 'std_pastData_30', 'std_pastData_40_min', 'std_pastData_40_max', 'std_pastData_40_mean', 'std_pastData_40', 'std_pastData_50_min', 'std_pastData_50_max', 'std_pastData_50_mean', 'std_pastData_50', 'zScore_pastData_5_min', 'zScore_pastData_5_max', 'zScore_pastData_5_mean', 'zScore_pastData_5', 'zScore_pastData_10_min', 'zScore_pastData_10_max', 'zScore_pastData_10_mean', 'zScore_pastData_10', 'zScore_pastData_15_min', 'zScore_pastData_15_max', 'zScore_pastData_15_mean', 'zScore_pastData_15', 'zScore_pastData_20_min', 'zScore_pastData_20_max', 'zScore_pastData_20_mean', 'zScore_pastData_20', 'zScore_pastData_25_min', 'zScore_pastData_25_max', 'zScore_pastData_25_mean', 'zScore_pastData_25', 'zScore_pastData_30_min', 'zScore_pastData_30_max', 'zScore_pastData_30_mean', 'zScore_pastData_30', 'zScore_pastData_40_min', 'zScore_pastData_40_max', 'zScore_pastData_40_mean', 'zScore_pastData_40', 'zScore_pastData_50_min', 'zScore_pastData_50_max', 'zScore_pastData_50_mean', 'zScore_pastData_50', 'ATR_3', 'ATR_5', 'ATR_7', 'ATR_10', 'ATR_15', 'ATR_20', 'ATR_25', 'profit_MTM_min', 'profit_MTM_max', 'profit_MTM_mean', 'profit_MTM', 'profit_MTM_Momentum_5_min', 'profit_MTM_Momentum_5_max', 'profit_MTM_Momentum_5_mean', 'profit_MTM_Momentum_5', 'profit_MTM_Momentum_10_min', 'profit_MTM_Momentum_10_max', 'profit_MTM_Momentum_10_mean', 'profit_MTM_Momentum_10', 'profit_MTM_Momentum_15_min', 'profit_MTM_Momentum_15_max', 'profit_MTM_Momentum_15_mean', 'profit_MTM_Momentum_15', 'profit_MTM_Momentum_20_min', 'profit_MTM_Momentum_20_max', 'profit_MTM_Momentum_20_mean', 'profit_MTM_Momentum_20', 'profit_MTM_Momentum_25_min', 'profit_MTM_Momentum_25_max', 'profit_MTM_Momentum_25_mean', 'profit_MTM_Momentum_25']
target = 'profitable'

trainDf = df[df.index.year < 2015]
testDf = df[df.index.year > 2014]

train_x = trainDf[features]
train_y = trainDf[[target]]

test_x = testDf[features]
test_y = testDf[[target]]

features = [features, target]
traintestData = [train_x,train_y, test_x, test_y]
algoData = ['sklearn.ensemble', 'RandomForestClassifier',"n_estimators = 120, n_jobs = 8, class_weight = 'balanced'"]
cutOff = 0.5
saveBestModel = './bestMomentumModel.pickle'

fs = FeatureSelection(features, traintestData, algoData, saveBestModel, cutOff = cutOff, sampling = 8)
