import numpy as np


def findSSE(Y, YHat):
    return np.sum((Y - YHat) ** 2)


def findMSE(Y, YHat):
    datasetCount = Y.shape[0]
    return findSSE(Y, YHat) / datasetCount


def findMAE(Y, YHat):
    datasetCount = Y.shape[0]
    return np.sum(np.abs(Y - YHat)) / datasetCount


def findMAPE(Y, YHat):
    datasetCount = Y.shape[0]
    return np.sum(np.abs(Y - YHat)) * 100 / datasetCount


def findEntropy(Y, YHat):
    return np.sum(-Y * np.log(YHat))


def findBinaryClass(Y, YHat):
    datasetCount = Y.shape[0]
    _Y = np.round(Y, 0)
    _YHat = np.round(YHat, 0)
    return 100 * np.sum(_Y != _YHat) / datasetCount


def findMultiClass(Y, YHat):
    datasetCount = Y.shape[0]
    Y_Argmax = np.argmax(Y, axis=1)
    YHat_Argmax = np.argmax(YHat, axis=1)
    return 100 * np.sum(Y_Argmax != YHat_Argmax) / datasetCount


def loss(Y, YHat, TypeOfLoss: str):
    if TypeOfLoss == 'SSE':
        loss = findSSE(Y, YHat)
    elif TypeOfLoss == 'MSE':
        loss = findMSE(Y, YHat)
    elif TypeOfLoss == 'MAE':
        loss = findMAE(Y, YHat)
    elif TypeOfLoss == 'MAPE':
        loss = findMAPE(Y, YHat)
    elif TypeOfLoss == 'Entropy':
        loss = findEntropy(Y, YHat)
    elif TypeOfLoss == 'Binary':
        loss = findBinaryClass(Y, YHat)
    elif TypeOfLoss == 'Multiclass':
        loss = findMultiClass(Y, YHat)
    else:
        raise Exception("Not supported loss function")
    return loss
