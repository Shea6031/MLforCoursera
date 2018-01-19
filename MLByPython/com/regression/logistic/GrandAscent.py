import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
    pass


def sigmoid(inx):
    return 1/(1+math.exp(inx))
    pass


def gradAscent(dataMatIn, classLabels):
    '''

    :param dataMatIn: 二维numpy，每行代表不同的样本值，每列表示不同的特征
    :param classLabels: 类别标签，
    :return:
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.T * error
    return weights
    pass

def plotBestFit(wei):
    weights = wei.getA()
    dataMat,labelMat =loadDataSet();
    dataArr = np.array(dataMat)
    n = np.shape(dataArr[0])
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()