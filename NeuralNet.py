class NeuralNet:
    import numpy as np

    import Layer
    import Loss
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

    def getOutputSize(self, layerIndex: int):
        return self.layerList[layerIndex][0]

    def forward(self, A0):
        Z = []
        A = []
        for i in range(len(self.layerList)):
            Z_i, A_i = Layer.layer(
                A0, i, self.weightList, self.biasList, self.layerList)
            Z.append(Z_i)
            A.append(A_i)
        return Z, A

    def backward(self):
        weightAdj = []
        biasAdj = []

    def build(self):
        for i in range(len(self.layerList) - 1):
            currOutputSize = self.getOutputSize(i)
            nextOutputSize = self.getOutputSize(i + 1)

            weightTmp = np.random.randn(
                currOutputSize, nextOutputSize) / np.sqrt(nextOutputSize)
            biasTmp = np.random.randn(
                1, nextOutputSize) / np.sqrt(nextOutputSize)

            self.weightList.append(weightTmp)
            self.weightList.append(biasTmp)

    def train(self, X, Y, epoch, leaningRate=0.01, lossFunction='Entropy'):
        layerCount = len(self.layerList)
        datasetCount = X.shape[0]
        for i in range(epoch):
            Z, A = self.foward(X)
            loss = Loss.loss(Y, A[-1], lossFunction)
            self.errorList.append(loss)
            weightAdj, biasAdj = 1, 1

    def predict(self, A0):
        Z, A = self.forward(A0)
        return A

    def loadModel(self, weight, bias, layer):
        self.weightList = weight
        self.biasList = bias
        self.layerList = layer

    def dumpModel(self):
        weight = self.weightList
        bias = self.biasList
        layer = self.layerList
        return weight, bias, layer
