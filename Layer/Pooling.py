import numpy as np


class Pooling:
    def __init__(self, kernelSize=(2, 2), stride=2, mode='max'):
        self.paras = {
            'kernelSize': kernelSize,
            'stride': stride,
            'mode': mode
        }
        self.cache = {}
        self.hasW = False

    def hasWeight(self):
        return self.hasW

    def forward(self, A0, saveCache=False):
        (num, old_h, old_w, old_c) = A0.shape
        kernel_h, kernel_w = self.paras['kernelSize']

        new_h = int(1 + (old_h - kernel_h) / self.paras['stride'])
        new_w = int(1 + (old_w - kernel_w) / self.paras['stride'])
        new_c = old_c

        A = np.zeros(A0.shape)

        for i in range(num):
            for h in range(new_h):
                for w in range(new_w):
                    vStart = self.paras['stride'] * h
                    vEnd = vStart + kernel_h
                    hStart = self.paras['stride'] * w
                    hEnd = hStart + kernel_w
                    for c in range(new_c):
                        if self.paras['mode'] == 'average':
                            A[i, h,  w, c] = np.mean(
                                A0[i, vStart:vEnd, hStart:hEnd, c])
                        elif self.paras['mode'] == 'max':
                            A[i, h, w, c] = np.max(
                                A0[i, vStart:vEnd, hStart:hEnd, c])

        if saveCache:
            self.cache['A'] = A0

        return A

    def disValue(self, dZ, shape):
        new_h, new_w = shape
        avg = 1 / (new_h * new_w)
        return np.ones(shape) * dZ * avg

    def mask(self, x):
        return x == np.max(x)

    def backward(self, dA):
        A = self.cache['A']
        kernel_h, kernel_w = self.paras['kernelSize']
        (num, old_h, old_w, old_c) = A.shape
        num_n, new_h, new_w, new_c = dA.shape

        dA0 = np.zeros(A.shape)

        for i in range(num):
            a = A[i]
            for h in range(new_h):
                for w in range(new_w):
                    vStart = self.paras['stride'] * h
                    vEnd = vStart + kernel_h
                    hStart = self.paras['stride'] * w
                    hEnd = hStart + kernel_w
                    for c in range(new_c):
                        if self.paras['mode'] == 'avarage':
                            da = dA[i, h, w, c]
                            dA0[i, vStart:vEnd, hStart:hEnd,
                                c] += self.disValue(da, self.paras['kernelSize'])
                        elif self.paras['mode'] == 'max':
                            aSlice = a[vStart:vEnd, hStart:hEnd, c]
                            mask = self.mask(aSlice)
                            dA0[i, vStart:vEnd, hStart:hEnd,
                                c] += dA[i, h, w, c] * mask

        return dA0
