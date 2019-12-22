import numpy as np
from NeuralNetNumpy.Activation import *


class Layer():
    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize

    def forward(self, A0):
        pass

    def backward(self, dA):
        pass

    def adjustParams(self, dW, dB, learningRate):
        pass


class Flatten(Layer):
    def __init__(self, inputSize=None):
        self.inputSize = inputSize

    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize
        if self.inputSize is None:
            raise ValueError("Input Size is not set!")

        self.outputSize = np.prod(self.inputSize)

    def forward(self, A0):
        # A0 -> !Z -> A
        A = np.ravel(A0).reshape(A0.shape[0], -1)
        return A.T

    def backward(self, dA):
        # dA -> !dZ -> dA0
        a = dA.shape[1]
        shape = tuple([a] + [i for i in self.inputSize])
        dA0 = dA.reshape(shape)
        return dA0, None, None

    def adjustParams(self, dW, dB, learningRate):
        pass


class Dense(Layer):
    def __init__(self, outputSize: int, Activation: Activation, inputSize=None):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = Activation

    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize
        if self.inputSize is None:
            raise ValueError("Input Size is not set!")
        W, B = self.initW_B((self.inputSize, self.outputSize))

        self.weight = W
        self.bias = B

    def forward(self, A0):
        # A0 -> Z -> A
        Z = np.dot(self.weight, A0) + self.bias
        A = self.activation.cal(Z)

        self.A0 = A0
        self.Z = Z
        self.A = A
        return A

    def backward(self, dA):
        # dA -> dZ -> dA0
        dZ = self.activation.diff(dA)
        count = dA.shape[1]

        dA0 = np.dot(self.weight.T, dZ)
        dW = np.dot(dZ, self.A0.T) / count
        dB = np.sum(dZ, axis=1, keepdims=True)
        return dA0, dW, dB

    def adjustParams(self, dW, dB, learningRate):
        self.weight -= learningRate * (dW + self.weight)
        self.bias -= learningRate * (dB + self.bias)

    def getFans(self, shape):
        fanIn = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fanOut = shape[1] if len(shape) == 2 else shape[1]
        return fanIn, fanOut

    def initW_B(self, shape):
        fanIn, fanOut = self.getFans(shape)
        scale = np.sqrt(2. / fanIn)
        shape = (fanOut, fanIn) if len(shape) == 2 else shape
        biasShape = (fanOut, 1) if len(shape) == 2 else (1, 1, 1, shape[3])

        normal = np.random.normal(0, scale, size=shape)
        uniform = np.random.uniform(-0.05, 0.05, size=biasShape)
        return normal, uniform


class Conv(Layer):
    def __init__(self, kernelCount: int, activation: Activation, kernelSize=(3, 3), padding='valid', stride=1, inputSize=None):
        supportedPadding = ['valid', 'same']
        if padding not in supportedPadding:
            raise TypeError("\'{}\' padding not founded. Only {} is supported".format(
                padding, supportedPadding))
        self.kernelCount = kernelCount
        self.activation = activation
        self.kernelSize = kernelSize
        self.padding = padding
        self.stride = stride
        self.inputSize = inputSize

    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize
        if self.inputSize is None:
            raise ValueError("Input Size is not set!")

        old_H, old_W, old_C = self.inputSize
        kernelSize_H, kernelSize_W = self.kernelSize

        if self.padding == 'same':
            pad_H = int(((old_H - 1) * self.stride + kernelSize_H - old_H) / 2)
            pad_W = int(((old_W - 1) * self.stride + kernelSize_W - old_W) / 2)
            new_H = old_H
            new_W = old_W
        elif self.padding == 'valid':
            pad_H = 0
            pad_W = 0
            new_H = int((old_H - kernelSize_H) / self.stride) + 1
            new_W = int((old_W - kernelSize_W) / self.stride) + 1
        else:
            raise TypeError("\'{}\' padding not founded.".format(self.padding))

        self.paddingSize = (pad_H, pad_W)
        self.outputSize = (new_H, new_W, self.kernelCount)

        shape = (kernelSize_H, kernelSize_W, old_C, self.kernelCount)
        W, B = self.initW_B(shape)

        self.weight = W
        self.bias = B

    def forward(self, A0):
        kernelSize_H, kernelSize_W = self.kernelSize
        pad_H, pad_W = self.paddingSize
        new_H, new_W, new_C = self.outputSize

        Z = np.zeros((len(A0), new_H, new_W, new_C))
        A0 = self.padInput(A0)

        for i in range(len(A0)):
            x = A0[i]
            for h in range(new_H):
                for w in range(new_W):

                    vStart = self.stride * h
                    vEnd = vStart + kernelSize_H

                    hStart = self.stride * w
                    hEnd = hStart + kernelSize_W

                    for c in range(new_C):
                        xSlice = x[vStart:vEnd, hStart:hEnd, :]
                        Z[i, h, w, c] = self.singleStep(
                            xSlice, self.weight[:, :, :, c], self.bias[:, :, :, c])

        A = self.activation.cal(Z)

        self.Z = Z
        self.A = A
        self.A0 = A0

        return A

    def backward(self, dA):
        # dA -> dZ -> dA0
        kernelSize_H, kernelSize_W = self.kernelSize
        pad_H, pad_W = self.paddingSize
        old_H, old_W, old_C = self.inputSize
        new_H, new_W, new_C = self.outputSize

        A0 = self.A0

        dZ = self.activation.diff(dA)

        dA0 = np.zeros((len(dA), old_H, old_W, old_C))
        dW = np.zeros_like(self.weight)
        dB = np.zeros_like(self.bias)

        A0_pad = self.padInput(A0)
        dA0_pad = self.padInput(dA0)

        for i in range(dZ.shape[0]):
            a_pad = A0_pad[i]
            da_pad = dA0_pad[i]

            for h in range(old_H):
                for w in range(old_W):

                    vStart = self.stride * h
                    vEnd = vStart + kernelSize_H

                    hStart = self.stride * w
                    hEnd = hStart + kernelSize_W

                    for c in range(new_C):
                        aSlice = a_pad[vStart:vEnd, hStart:hEnd, :]
                        da_pad[vStart:vEnd, hStart:hEnd,
                               :] += self.weight[:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += aSlice * dZ[i, h, w, c]
                        dB[:, :, :, c] += dZ[i, h, w, c]
            dA0[i, :, :, :] = da_pad[pad_H:-pad_H, pad_W:-pad_W, :]

        return dA0, dW, dB

    def adjustParams(self, dW, dB, learningRate):
        self.weight -= learningRate * (dW + self.weight)
        self.bias -= learningRate * (dB + self.bias)

    def singleStep(self, x, W, B):
        return np.sum(np.multiply(x, W) + float(B))

    def getFans(self, shape):
        fanIn = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fanOut = shape[1] if len(shape) == 2 else shape[1]
        return fanIn, fanOut

    def initW_B(self, shape):
        fanIn, fanOut = self.getFans(shape)
        scale = np.sqrt(2. / fanIn)
        shape = (fanIn, fanOut) if len(shape) == 2 else shape
        biasShape = (fanOut, 1) if len(shape) == 2 else (1, 1, 1, shape[3])

        normal = np.random.normal(0, scale, size=shape)
        uniform = np.random.uniform(-0.05, 0.05, size=biasShape)
        return normal, uniform

    def padInput(self, X):
        pad = self.paddingSize
        Y = np.pad(X, ((0, 0), (pad[0], pad[0]),
                       (pad[1], pad[1]), (0, 0)), 'constant')
        return Y


class MaxPool(Layer):
    def __init__(self, kernelSize=(2, 2), stride=2, inputSize=None):
        self.kernelSize = kernelSize
        self.stride = stride
        self.inputSize = inputSize

    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize
        if self.inputSize is None:
            raise ValueError("Input Size is not set!")

        old_H, old_W, old_C = self.inputSize
        kernel_H, kernel_W = self.kernelSize

        new_H = int(1 + (old_H - kernel_H) / self.stride)
        new_W = int(1 + (old_H - kernel_W) / self.stride)
        new_C = old_C

        self.outputSize = (new_H, new_W, new_C)

    def forward(self, A0):
        # A0 -> !Z -> A
        new_H, new_W, new_C = self.outputSize
        A = np.zeros((len(A0), new_H, new_W, new_C))

        for i in range(len(A0)):
            for h in range(new_H):
                for w in range(new_W):

                    vStart = h * self.stride
                    vEnd = vStart + self.kernelSize[0]

                    hStart = w * self.stride
                    hEnd = hStart + self.kernelSize[1]

                    for c in range(new_C):
                        A[i, h, w, c] = np.max(
                            A0[i][vStart:vEnd, hStart:hEnd, c])

        self.A0 = A0
        self.A = A

        return A

    def backward(self, dA):
        # dA -> !dZ -> dA0
        old_H, old_W, old_C = self.inputSize
        new_H, new_W, new_C = self.outputSize

        A0 = self.A0

        dA0 = np.zeros((len(dA), old_H, old_W, old_C))

        for i, data in enumerate(dA):
            a = A0[i]
            for h in range(new_H):
                for w in range(new_W):

                    vStart = h * self.stride
                    vEnd = vStart + self.kernelSize[0]

                    hStart = w * self.stride
                    hEnd = hStart + self.kernelSize[1]

                    for c in range(new_C):
                        aSlice = a[vStart:vEnd, hStart:hEnd, c]
                        mask = self.createMask(aSlice)
                        dA0[i, vStart:vEnd, hStart:hEnd,
                            c] += dA[i][h, w, c] * mask

        return dA0, None, None

    def adjustParams(self, dW, dB, learningRate):
        pass

    def createMask(self, x):
        return x == np.max(x)


""" ちゃろー☆
    &Wggggggr                   ||gMT||||||||||@M*`                  @          *g
         ,@"                 ,|,@T||||||||yg@M'             ,       ]jL           "W
      ,;$R                 ||,@*|||||ill@M*'             ,iL       ,@}@
    l||$F                ||,@F,|illm&$@"|      ,     ,yllT` ;L     $|#$
    '`#F               |||g@@&M*'||@F|||| ||   'L.ill|@F` ;l|L    $@@*$       ],
     $F   |          ,||,@`     ,@"|||||||||||lLjL,g@F' ;@F|T    $$WMT$L       $
    @M   |          |  #F%wwg,,@'|||||||||||||||@M@&` jg@||lL   gF||||$L        @
    @   |    :,    |||$Lggggg@$$@g,|||||||,g@M" ,@lL|@@F'l|L   #F|  ||$         ]L
    `  |       '*Wgg$@@@@@@@$$$$$$$g@@@M*'`   ,@||y#L]F|lL|` ,@L|   |}$          @
      |         | "$@@@@@@@NMMMMN@@@@@@@@g  ,@|,@F` $F||||L gK ||||||j@          $
               |  |@@@M*"|||,gggggg,|*%@@@@$@&*"W,,@|lL|||;@`    ||||$F        | ]
     |    |L      $@@'   g@@@M*$$$$@@@g,*M@@@g   ]ML||lW|@"       "||$        .L ]F
    |     jF  |  $$@L  g@@@@@,,g@@@$$$$@g||j@@@g@T|||Lg&$          ']F        l` ]k
          jF |  ]M$F ,@@@@@@@@@@@@@@$@$$@|||'%@$|||,gM'  `          @        gW  $k
          jF    & $ j@@@@@@@@@@@@@@@@$@$@ '';&*$g@M"               $L    ,wMllL j@TL
          |@   jF M $@$$$$@@@@$$$$@@@$$@@  gLgM*T                 1F,,W*'` l$F  $|lL
           $   $L 'j$T|l&@@@@@@$$$$@NN@@F                       +@F'     |g@F  $FllL
           ]@| $    L  ' j$$$$$@@M'    &                        gF     ,g@]F  $MlllL
    ,,ggwwg $L $ |  "@    "*MMMMT                              g"|  |,@M'|&||$$|lllL
    '` ,gM' j@,$||   '*,       ||L,,                         ,@li|l@M'  ,F|y@$||||||
    g@*`||  j@&$L|  ||'     ,,|,g@M"                     wM"lLw&*$,,gg@@@$@$@|||||||
    &gg,     $Lj@L ,|L |      ''`                          `,gg@@@@@@@@@@@@@||||||||
       '"%g, ]W||&|||  |||'                             g@@@@@@@@NMM*""'`;&|||||||||
    @F**M*'` |$|||||||||'             ,,ggg,,          j%@@$L'`         gL|||||||||@
    |$L '&L   jg||||||'            ,@F" ||||||%g         '*%@@@g,     g&||||||||g&M|
    l|jg  ]g   $L||||;||'        gF`   |||||||||*g           '%@@@@ggM`||||,g@M|#T||
    g@NM%g j@, '%g|||'`        ,@`       ||||||||j@g            "%N@@@@@F"'|||gM|,g@
    '%||||%gl&@,'%@g,          @`          ||||||j$'            ,L '$@F|||||yM|g@T$L
      "W|||||&$l@,%l%@g       |F            |||||$F             || j@`| |,g$p&" |@"
    '   jw||||||M%@@gj@@@g,   '@             |||#"          |   || |$L||`'||j$||@`
         '%||||||||&|1&$@@@@gg "W            |gF            || |||||'$L   ||@L;@
        '  jw|||||||%g||j$@MM%@@@g,     ,,,wM"              '|||||,g@F`   |$LgP
            "@|||||||jg|T%@||||||%%@@@g$,`                  ,,gg@@M'     |jFgF
             '&|||||||l@||"%@|||||@T|#MM%N@@@@@@@@@@@@@@@@NM%F''}@       |$$L
               $||||||||@L||'%g|@$gM||||||||||||j|||$F|||||l'$L|}$L      $@$
     ,          &    |||"@ ''|j@T||||||||||||||l@|lj&|||||lL  jL}$@    |$@ $L     yF
"""
