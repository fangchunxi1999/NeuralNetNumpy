class Loss:
    import numpy as np

    def findSSE(self, Y, YHat):
        return np.sum((Y - YHat) ** 2)

    def findMSE(self, Y, YHat):
        datasetCount = Y.shape[0]
        return self.findSSE(Y, YHat) / datasetCount

    def findMAE(self, Y, YHat):
        datasetCount = Y.shape[0]
        return np.sum(np.abs(Y - YHat)) / datasetCount

    def findMAPE(self, Y, YHat):
        datasetCount = Y.shape[0]
        return np.sum(np.abs(Y - YHat)) * 100 / datasetCount

    def findEntropy(self, Y, YHat):
        return np.sum(-Y * np.log(YHat))

    def findBinaryClass(self, Y, YHat):
        datasetCount = Y.shape[0]
        _Y = np.round(Y, 0)
        _YHat = np.round(YHat, 0)
        return 100 * np.sum(_Y != _YHat) / datasetCount

    def findMultiClass(self, Y, YHat):
        datasetCount = Y.shape[0]
        Y_Argmax = np.argmax(Y, axis=1)
        YHat_Argmax = np.argmax(YHat, axis=1)
        return 100 * np.sum(Y_Argmax != YHat_Argmax) / datasetCount

    def loss(self, Y, YHat, TypeOfLoss: str):
        if TypeOfLoss == 'SSE':
            loss = self.findSSE(Y, YHat)
        elif TypeOfLoss == 'MSE':
            loss = self.findMSE(Y, YHat)
        elif TypeOfLoss == 'MAE':
            loss = self.findMAE(Y, YHat)
        elif TypeOfLoss == 'MAPE':
            loss = self.findMAPE(Y, YHat)
        elif TypeOfLoss == 'Entropy':
            loss = self.findEntropy(Y, YHat)
        elif TypeOfLoss == 'Binary':
            loss = self.findBinaryClass(Y, YHat)
        elif TypeOfLoss == 'Multiclass':
            loss = self.findMultiClass(Y, YHat)
        else:
            raise Exception("Not supported loss function")

        return loss
