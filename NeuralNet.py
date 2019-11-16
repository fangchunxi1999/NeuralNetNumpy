import numpy as np
import NeuralNetNumpy.Layer as Layer
import NeuralNetNumpy.Loss as Loss
import NeuralNetNumpy.Activation as Activation
from random import shuffle


class NeuralNet:

    def __init__(self):
        self.weightList = []
        self.biasList = []
        self.errorList = []
        self.layerList = []

    def addLayer(self, layer: tuple):
        self.layerList.append(layer)

    def forward(self, A0):
        Z = []
        A = []
        for i in range(len(self.layerList)):
            if len(A) == 0:
                A_i = A0
            Z_i, A_i = Layer.layer(
                A_i, i, self.weightList, self.biasList, self.layerList)
            Z.append(Z_i)
            A.append(A_i)
        return Z, A

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

    def build(self):
        self.weightList = []
        self.biasList = []
        self.errorList = []
        for i in range(len(self.layerList) - 1):
            currOutputSize = self.getOutputSize(i)
            nextOutputSize = self.getOutputSize(i + 1)

            weightTmp = np.random.randn(
                currOutputSize, nextOutputSize) / np.sqrt(nextOutputSize)
            biasTmp = np.random.randn(
                nextOutputSize) / np.sqrt(nextOutputSize)

            self.weightList.append(weightTmp)
            self.biasList.append(biasTmp)

    def train(self, X: np.ndarray, Y: np.ndarray, epoch: int, leaningRate=0.01, batchSize=32, lossFunction='Entropy'):
        layerCount = len(self.layerList)
        datasetCount = len(X)

        nBatch = int(datasetCount / batchSize)

        for i in range(epoch):
            print("Iteartion: {}/{} ".format(i + 1, epoch), end='')

            loss = 0.0
            indices = np.random.permutation(datasetCount)
            X = X[indices]
            Y = Y[indices]

            for j in range(0, datasetCount, batchSize):
                #print(".", end='')
                X_i = X[i:i + batchSize]
                Y_i = Y[i:i + batchSize]

                Z, A = self.forward(X_i)
                loss = loss + Loss.loss(Y_i, A[-1], lossFunction)

                weightAdj, biasAdj = self.backward(Y_i, Z, A)
                for j in range(layerCount - 1):
                    self.weightList[j] = self.weightList[j] + \
                        (leaningRate / datasetCount) * weightAdj[j]
                    self.biasList[j] = self.biasList[j] + \
                        (leaningRate / datasetCount) * biasAdj[j]

            self.errorList.append(loss)

            print("(loss: {})".format(loss))

    def predict(self, A0):
        Z, A = self.forward(A0)
        return A[-1]

    def getOutputSize(self, layerIndex: int):
        return self.layerList[layerIndex][0]

    def loadModel(self, weight, bias, layer):
        self.weightList = weight
        self.biasList = bias
        self.layerList = layer

    def dumpModel(self):
        weight = self.weightList
        bias = self.biasList
        layer = self.layerList
        return weight, bias, layer
