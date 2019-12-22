import numpy as np

from NeuralNetNumpy.Util.Initalizer import glorot_uniform
from NeuralNetNumpy.Util.Util import pad_inputs


class Conv2D:
    def __init__(self, kernelCount: int, kernelSize=(3, 3), padding='valid', stride=1):
        self.paras = {
            'kernelCount': kernelCount,
            'padding': padding,
            'kernelSize': kernelSize,
            'stride': stride
        }
        self.cache = {}
        self.grade = {}

        self.hasW = True

    def hasWeight(self):
        return self.hasW

    def singleStep(self, input, W, b):
        return np.sum(np.multiply(input, W)) + float(b)

    def forward(self, A0, saveCache=False):
        (num, old_h, old_w, old_c) = A0.shape
        kernel_h, kernel_w = self.paras['kernelSize']

        if 'W' not in self.paras:
            shape = (kernel_h, kernel_w, old_c, self.paras['kernelCount'])
            W, b = glorot_uniform(shape=shape)
            self.paras['W'] = W
            self.paras['b'] = b

        if self.paras['padding'] == 'same':
            pad_h = int(
                ((old_h - 1) * self.paras['stride'] + kernel_h - old_h) / 2)
            pad_w = int(
                ((old_w - 1) * self.paras['stride'] + kernel_w - old_w) / 2)
            new_h = old_h
            new_w = old_w
        elif self.paras['padding'] == 'valid':
            pad_h = 0
            pad_w = 0
            new_h = int((old_h - kernel_h) / self.paras['stride'] + 1)
            new_w = int((old_w - kernel_w) / self.paras['stride'] + 1)

        self.paras['pad_h'] = pad_h
        self.paras['pad_w'] = pad_w

        Z = np.zeros((num, new_h, new_w, self.paras['kernelCount']))

        A0_pad = pad_inputs(A0, (pad_h, pad_w))

        for i in range(num):
            x = A0_pad[i]
            for h in range(new_h):
                for w in range(new_w):
                    vStart = self.paras['stride'] * h
                    vEnd = vStart + kernel_h
                    hStart = self.paras['stride'] * w
                    hEnd = hStart + kernel_w
                    for c in range(self.paras['kernelCount']):
                        xSlice = x[vStart:vEnd, hStart:hEnd, :]
                        Z[i, h, w, c] = self.singleStep(
                            xSlice, self.paras['W'][:, :, :, c], self.paras['b'][:, :, :, c])

        if saveCache:
            self.cache['A'] = A0

        return Z

    def backward(self, dZ):
        A = self.cache['A']
        kernel_h, kernel_w = self.paras['kernelSize']
        pad_h = self.paras['pad_h']
        pad_w = self.paras['pad_w']

        (num, old_h, old_w, old_c) = A.shape

        dA0 = np.zeros((A.shape))

        self.grade['dW'] = np.zeros_like(self.paras['W'])
        self.grade['db'] = np.zeros_like(self.paras['b'])

        A_pad = pad_inputs(A(pad_h, pad_w))
        dA0_pad = pad_inputs(dA0, (pad_h, pad_w))

        for i in range(num):
            a_pad = A_pad[i]
            da_pad = dA0_pad[i]
            for h in range(old_h):
                for w in range(old_w):
                    vStart = self.paras['stride'] * h
                    vEnd = vStart + kernel_h
                    hStart = self.paras['stride'] * w
                    hEnd = hStart + kernel_w
                    for c in range(self.paras['kernelCount']):
                        aSlice = a_pad[vStart:vEnd, hStart:hEnd, :]
                        da_pad[vStart:vEnd, hStart:hEnd,
                               :] += self.paras['W'][:, :, :, c] * dZ[i, h, w, c]

                        self.grade['dW'][:, :, :, c] += aSlice * dZ[i, h, w, c]
                        self.grade['db'][:, :, :, c] += dZ[i, h, w, c]

            dA0[i, :, :, :] = da_pad[pad_h:-pad_h, pad_w:-pad_w, :]

        return dA0

    def applyGrads(self, learnRate=0.001, l2=1e-4):
        self.paras['W'] -= learnRate * \
            (self.grads['dW'] + l2 * self.paras['W'])
        self.paras['B'] -= learnRate * \
            (self.grads['db'] + l2 * self.paras['B'])
