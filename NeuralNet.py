import numpy as np

from NeuralNetNumpy.Util.Util import get_batches, evaluate


class NeuralNet:
    def __init__(self, *model, **kwargs):
        self.model = model
        self.loss = None
        self.numClass = 0

    def setNumClass(self, numClass: int):
        self.numClass = numClass

    def setLoss(self, loss):
        self.loss = loss

    def train(self, X, Y, epoch: int, loss, learnRate=0.001, l2=1e-4, batchSize=32):
        iter = 1
        for e in range(epoch):
            print("Epoch: ", e + 1)
            for i, (xBatch, yBatch) in enumerate(get_batches(X, Y, batch_size=batchSize)):
                batchPred = xBatch.copy()
                for num, layer in enumerate(self.model):
                    batchPred = layer.forward(batchPred, saveCache=True)

                dA = self.loss.compute_derivative(yBatch, batchPred)

                for layer in reversed(self.model):
                    dA = layer.backward(dA)

                for layer in self.model:
                    if layer.hasWeight():
                        layer.applyGrads(learnRate=learnRate, l2=l2)
        iter += batchSize

    def predict(self, X):
        batchSize = X.shape[0]
        prediction = np.zeros((1, X.shape[0]))

        numBatch = X.shape[0] // batchSize

        for batchNum, xBatch in enumerate(get_batches(X, batch_size=batchSize, shuffle=False)):
            batchPred = xBatch.copy()
            for layer in self.model:
                batchPred = layer.forward(batchPred)
            M, N = batchPred.shape

            if M != prediction.shape[0]:
                prediction = np.zeros((M, X.shape[0]))

            if batchNum <= numBatch - 1:
                prediction[:, batchNum *
                           batchSize:(batchNum + 1) * batchSize] = batchPred
            else:
                prediction[:, batchNum * batchSize:] = batchPred

        return prediction

    def evaluate(self, X, Y):
        prediction = self.predict(X)
        M, N = prediction.shape
        if (M, N) == Y.shape:
            return evaluate(Y, prediction)
        elif (N, M) == Y.shape:
            return evaluate(Y.T, prediction)

    def load(self, model):
        self.model = model

    def dump(self):
        return self.model


def createOnehot(self, Y):
    datasetCount = len(Y)
    classCount = len(np.unique(Y))
    onehot = np.zeros([datasetCount, classCount])
    for i in range(datasetCount):
        onehot[i, Y[i]] = 1
    return onehot
