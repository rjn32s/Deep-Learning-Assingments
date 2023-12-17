import numpy as np
from scipy import signal
def kaiming(n,fil):
  mean =0
  sdT =np.sqrt(2/n)
  val = np.random.normal(mean ,sdT , size=fil)
  return val

kaiming(1,(3,3))
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

data = np.load("/content/dataset.npz")
trainX, trainy, testX, testy, valX, valy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5']
print('Loaded: ', trainX.shape, len(trainy), testX.shape, len(testy), valX.shape, len(valy) )
def array_rsh(arr):
  l = []
  for i in arr:
    l.append(np.reshape( i, (224, 224,1) ))
  return np.array(l)

trainX = array_rsh(trainX)
#trainY = array_rsh(trainy)
testX = array_rsh(testX)
#testY = array_rsh(testy)
valX = array_rsh(valX)
#trainX = array_rsh(valy)


trainX = trainX/255.0
testX = testX/255.0
valX = valX/255.0
from keras.utils import np_utils
np.unique(trainy)



def label_generator(arr):
  stringlist = np.unique(arr)

  for ele in range(len(arr)):
    if arr[ele] == 'bonsai':
      arr[ele] = 0
    if arr[ele] == 'scorpion':
      arr[ele] = 1
    if arr[ele] == 'sunflower':
      arr[ele] = 2
  
  return arr


trainy = label_generator(trainy)
testy = label_generator(testy)
valy = label_generator(valy)
      
trainY = np_utils.to_categorical(trainy)
testY = np_utils.to_categorical(testy)
valY = np_utils.to_categorical(valy)
#y = np_utils.to_categorical(trainy)


q1=Convolutional((1,224,224),3,1)
output = q1.forward(trainX[21])

print(output.shape)
import matplotlib.pyplot as plt

plt.imshow(trainX[21][0,:,:],cmap = 'gray')
plt.show()