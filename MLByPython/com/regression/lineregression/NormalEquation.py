#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:46:14 2017

@author: xj2sgh
"""

"""
Dataset introduction: http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
Dataload: http://archive.ics.uci.edu/ml/machine-learning-databases/00294/
"""

import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def getData(filepath):
    df = pd.DataFrame()
    df = pd.read_excel(filepath, header=0)
    m, n = df.shape
    trainDf = df[:int(m * 0.7)]
    testDf = df[int(m * 0.7):]
    trainMat = trainDf.as_matrix(columns=None)
    testMat = testDf.as_matrix(columns=None)
    return trainMat, testMat

def normalEquation(trainMat):
    y_train = trainMat[:,-1:]
    m,n = y_train.shape
    one = np.mat(np.ones((m, 1)))
    x_train = np.hstack((one,trainMat[:, :-1]))
    theta = (x_train.T*x_train).I*x_train.T*y_train
    return theta

def test(testMat,theta):
    x_test = testMat[:,:-1]
    y_test = testMat[:,-1:]
    m, n = x_test.shape
    one = np.mat(np.ones((m, 1)))
    x_data = np.hstack((one, x_test))
    y_pre = x_data*theta
    err = (y_pre-y_test)/y_test
    return y_pre,err

if __name__ == '__main__':
    filepath = '/Users/xj2sgh/PycharmProjects/MachineLearningPractice/Dataset/CCPP/Folds5x2_pp.xlsx'
    trainData, testData = getData(filepath)
    theta = normalEquation(trainData)
    print(theta)
    preResult,err = test(testData,theta)
    print(preResult)
    plt.scatter(testData[:, -1:], np.asarray(preResult))
