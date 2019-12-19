import numpy as np
import NeuralNetNumpy.Loss as Loss
from NeuralNetNumpy.Layer import *
from NeuralNetNumpy.Activation import *
from random import shuffle


class NeuralNet:

    def __init__(self):
        self.errorList = []
        self.layerList = []

    def add(self, Layer: Layer):
        self.layerList.append(Layer)

    def build(self):
        self.errorList = []
        for i, l in enumerate(self.layerList):
            if i == 0:
                l.build()
                nextInputSize = l.outputSize
            else:
                l.build(nextInputSize)
                nextInputSize = l.outputSize

    def forward(self, A0):
        A = []
        A_i = A0
        for l in self.layerList:
            A_i = l.forward(A_i)
            A.append(A_i)
        return A

    def backward(self, Y, Z, A):
        layerCount = len(self.layerList)
        weightAdj = []
        biasAdj = []
        err = None
        for i in range(layerCount - 1, 0, -1):
            if i == layerCount - 1:
                delta = Y - A[i]
                diff = 1
            else:
                delta = np.dot(err, self.weightList[i].T)
                diff = Activation.activationDiff(
                    Z[i], A[i], self.layerList[i][1])
            err = delta * diff

            wAdj = np.dot(A[i - 1].T, err)
            bAdj = np.array(np.sum(err, axis=0))

            weightAdj.append(wAdj)
            biasAdj.append(bAdj)

        return weightAdj[::-1], biasAdj[::-1]

    def backward(self, Y, A):
        weightAdj = []
        biasAdj = []
        dA_i = Y - A[-1]
        for l in reversed(self.layerList):
            dA_i, dW_i, dB_i = l.backward(dA_i)
            weightAdj.append(dW_i)
            biasAdj.append(dB_i)

        return weightAdj[::-1], biasAdj[::-1]

    def createOnehot(self, Y):
        datasetCount = len(Y)
        classCount = len(np.unique(Y))
        onehot = np.zeros([datasetCount, classCount])
        for i in range(datasetCount):
            onehot[i, Y[i]] = 1
        return onehot

    def train(self, X: list, Y: list, epoch: int, learningRate=0.01, batchSize=32, lossFunction='Entropy'):
        datasetCount = len(X)

        nBatch = int(datasetCount / batchSize)

        for i in range(epoch):
            print("Iteartion: {}/{} ".format(i + 1, epoch), end='')

            loss = 0.0
            tmp = list(zip(X, Y))
            shuffle(tmp)
            X, Y = zip(*tmp)

            for j in range(0, datasetCount, batchSize):
                #print(".", end='')
                X_i = X[i:i + batchSize]
                Y_i = Y[i:i + batchSize]
                Y_i = self.createOnehot(Y_i)

                A = self.forward(X_i)
                loss = loss + Loss.loss(Y_i, A[-1], lossFunction)

                weightAdj, biasAdj = self.backward(Y_i, A)
                for i, l in enumerate(self.layerList):
                    l.adjustParams(
                        weightAdj[i], biasAdj[i], learningRate, datasetCount)

            self.errorList.append(loss)
            #learningRate = learningRate * 1 / (1 + lrDecay * epoch)
            print(" (loss: {})".format(loss))

    def predict(self, A0: list):
        return self.forward(A0)[-1]

    def loadModel(self, layer):
        self.layerList = layer

    def dumpModel(self):
        layer = self.layerList
        return layer
