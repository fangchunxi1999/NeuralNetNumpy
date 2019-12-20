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

    def adjustParams(self, dW, dB, learningRate, count):
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
        A0 = np.concatenate([i[np.newaxis] for i in A0])
        self.A = np.empty((0, self.outputSize), int)
        for i in A0:
            i = i.reshape(1, -1)
            self.A = np.vstack((self.A, i))
        return self.A

    def backward(self, dA):
        dA = [i.reshape(self.inputSize) for i in np.vsplit(dA, dA.shape[0])]
        return dA, None, None

    def adjustParams(self, dW, dB, learningRate, count):
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

        self.weight = np.random.randn(
            self.inputSize, self.outputSize) / np.sqrt(self.outputSize)
        self.bias = np.random.randn(self.outputSize) / np.sqrt(self.outputSize)

    def forward(self, A0):
        self.Z = np.dot(A0, self.weight) + self.bias
        self.A = self.activation.cal(self.Z)
        return self.A

    def backward(self, dA):
        dA0 = np.dot(dA, self.weight.T)
        dW = np.dot(self.Z.T, dA0).T
        dB = np.sum(dA, axis=0, keepdims=True)
        return dA0, dW, dB

    def adjustParams(self, dW, dB, learningRate, count):
        self.weight = self.weight + (learningRate / count) * dW
        self.bias = self.bias + (learningRate / count) * dB


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

        self.paddingSize = (pad_H, pad_W)
        self.outputSize = (new_H, new_W, self.kernelCount)

        shape = (kernelSize_H, kernelSize_W, old_C, self.kernelCount)
        self.weight, self.bias = self.initW_B(shape)

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
        uniform = np.random.uniform(-scale, scale, size=shape)
        return normal, uniform


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
        new_H, new_W, new_C = self.outputSize
        A = []

        for data in A0:
            tmp_A = np.zeros((new_H, new_W, new_C))
            for h in range(new_H):
                for w in range(new_W):

                    vStart = h * self.stride
                    vEnd = vStart + self.kernelSize[0]

                    hStart = w * self.stride
                    hEnd = hStart + self.kernelSize[1]

                    for c in range(new_C):
                        tmp_A[h, w, c] = np.max(
                            data[vStart:vEnd, hStart:hEnd, c])
            A = A + [tmp_A]

        self.Z = A0
        self.A = A

        return self.A

    def backward(self, dA):
        old_H, old_W, old_C = self.inputSize
        new_H, new_W, new_C = self.outputSize

        dA0 = []

        for i, data in enumerate(dA):
            a = self.Z[i]
            tmp_dA0 = np.zeros((old_H, old_W, old_C))

            for h in range(new_H):
                for w in range(new_W):

                    vStart = h * self.stride
                    vEnd = vStart + self.kernelSize[0]

                    hStart = w * self.stride
                    hEnd = hStart + self.kernelSize[1]

                    for c in range(new_C):
                        aSlice = a[vStart:vEnd, hStart:hEnd, c]
                        mask = self.createMask(aSlice)
                        tmp_dA0[vStart:vEnd, hStart:hEnd,
                                c] += data[h, w, c] * mask
            dA0 = dA0 + [tmp_dA0]

        return dA0

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
