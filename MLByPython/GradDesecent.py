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
import  matplotlib.pyplot as plt


def getData(filepath):
    df = pd.DataFrame()
    df = pd.read_excel(filepath,header = 0)
    m,n = df.shape
    trainDf = df[:int(m*0.7)]
    testDf = df[int(m*0.7):]
    trainMat = trainDf.as_matrix(columns=None)
    testMat = testDf.as_matrix(columns=None)
    return trainMat,testMat

def lineRegression(dataMatrixIn,theta):
    return np.dot(dataMatrixIn,theta)


def gradDescent(dataMatrixIn,maxCycles,epsilon):
    dataMat = dataMatrixIn.copy()
    m,n = dataMat.shape
    theta = np.random.rand(n,1)
    err = np.ones((n,1))
    h = np.zeros((m,1))
    one = np.mat(np.ones((m, 1)))
    x = np.hstack((one, dataMat[:, :-1]))
    y = dataMat[:, -1:]
    alpha = 0.0000005
    count = 0
    while count < maxCycles:
        count += 1
        h = lineRegression(x, theta)
        dif = y-h    
        sumValue = (np.multiply(dif, x)).sum(axis = 0)
        theta += alpha * (1 / m) *sumValue.T
        if np.linalg.norm(theta - err) < epsilon:
            break
    return theta

def predict(w,dataMatrixIn):
    dataMat = dataMatrixIn.copy()
    m,n = dataMat.shape
    one = np.mat(np.ones((m, 1)))
    data = np.hstack((one, dataMat[:, :-1]))
    result = lineRegression(data,w)
    return result

def errCompute(preValue,realValue):
    error = abs(preValue-realValue)/realValue
    return error


if __name__ == '__main__':
    filepath = '/Users/xj2sgh/PycharmProjects/MachineLearningPractice/Dataset/CCPP/Folds5x2_pp.xlsx'
    trainData,testData = getData(filepath)
    maxCyles = 100
    epsilon = 0.05
    theta= gradDescent(testData,maxCyles,epsilon)
    preResult = predict(theta,testData)
    plt.scatter(testData[:,-1:],np.asarray(preResult))
    