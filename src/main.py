import neuralNetwork
import numpy as np

import matplotlib.pyplot as mp
import scipy.misc
import os.path as pf
from PIL import Image
import image
mp.rcParams['toolbar'] = 'None'

#Mnist
data_file = open("../data/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()



#Values:
inputNodes = 784
hiddenNodes = 100
outputNodes = 10
learningRate = 0.3



#Imported Img
for i in range(100):
    if(pf.isfile('../photos/{}.jpg'.format(i))==1):
        img_array = scipy.misc.imread('../photos/{}.jpg'.format(i),flatten=True)
        img_data = 255.0 - img_array.reshape(784)
        fixed_color = 255.0 - img_array
        img_data = (img_data/255.0*0.99) + 0.01

        nw = neuralNetwork.createNeuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate)



        mp.imshow(fixed_color,cmap='Greys',interpolation='None')
        mp.text(1,1,"Computer Guess: "+str(np.argmax(nw.query(img_data))))
        mp.show()
