import numpy as np
import Layer
import Loss
import Activation


class NeuralNet:

    # Z = sum(W * X) + B , X = A
    # A = relu(Z)
    # read W,B save A,Z

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
        for i in range(len(self.layerList) - 1):
            currOutputSize = self.getOutputSize(i)
            nextOutputSize = self.getOutputSize(i + 1)

            weightTmp = np.random.randn(
                currOutputSize, nextOutputSize) / np.sqrt(nextOutputSize)
            biasTmp = np.random.randn(
                nextOutputSize) / np.sqrt(nextOutputSize)

            self.weightList.append(weightTmp)
            self.biasList.append(biasTmp)

    def train(self, X, Y, epoch, leaningRate=0.01, lossFunction='Entropy'):
        layerCount = len(self.layerList)
        datasetCount = len(X)
        for i in range(epoch):
            Z, A = self.forward(X)
            loss = Loss.loss(Y, A[-1], lossFunction)
            self.errorList.append(loss)
            weightAdj, biasAdj = self.backward(Y, Z, A)
            for j in range(layerCount - 1):
                self.weightList[j] = self.weightList[j] + \
                    (leaningRate / datasetCount) * weightAdj[j]
                self.biasList[j] = self.biasList[j] + \
                    (leaningRate / datasetCount) * biasAdj[j]
            print("Iteartion: {}/{}".format(i, epoch))

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
