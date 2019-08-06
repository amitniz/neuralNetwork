import numpy
import scipy.special
import os.path as pf
class createNeuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #Set num of nodes in each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #Learning rate
        self.lr = learningrate

        #Setting the Weights

        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        if(pf.isfile("data/wih.npy")==1):
            self.wih = numpy.load("data/wih.npy")
        if(pf.isfile("data/who.npy")==1):
            self.who = numpy.load("data/who.npy")


        #Sigmoid function

        self.activationFunction = lambda x: scipy.special.expit(x)


    def train(self,inputsList,targetsList):
        #Convert the inputs into an Array
        inputs = numpy.array(inputsList,ndmin=2).T
        targets = numpy.array(targetsList,ndmin=2).T


        #Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih,inputs)
        #Calculate the outputs of the hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        #Calculate signals into the output layer
        finalInputs = numpy.dot(self.who,hiddenOutputs)
        #Calculate the final output
        finalOutputs = self.activationFunction(finalInputs)

        #Calculate the output errors
        outputErrors = targets - finalOutputs
        #Calculate the hidden layer errors
        hiddenErrors = numpy.dot(self.who.T,outputErrors)

        #Update the Weights between the hidden and the output layers
        self.who += self.lr * numpy.dot((outputErrors*finalOutputs*(1.0-finalOutputs)),numpy.transpose(hiddenOutputs))
        #Update the weights between the input and the hidden layers
        self.wih += self.lr * numpy.dot((hiddenErrors*hiddenOutputs*(1.0-hiddenOutputs)),numpy.transpose(inputs))

        #Save the new weights
        numpy.save('data/who.npy',self.who)
        numpy.save('data/wih.npy',self.wih)

        return numpy.argmax(finalOutputs)

    def train(self, inputsList):
        # Convert the inputs into an Array
        inputs = numpy.array(inputsList, ndmin=2).T
        targets = int(input("What is the Number?"))

        # Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        # Calculate the outputs of the hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        # Calculate signals into the output layer
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        # Calculate the final output
        finalOutputs = self.activationFunction(finalInputs)

        # Calculate the output errors
        outputErrors = targets - finalOutputs
        # Calculate the hidden layer errors
        hiddenErrors = numpy.dot(self.who.T, outputErrors)

        # Update the Weights between the hidden and the output layers
        self.who += self.lr * numpy.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)),
                                        numpy.transpose(hiddenOutputs))
        # Update the weights between the input and the hidden layers
        self.wih += self.lr * numpy.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(inputs))

        # Save the new weights
        numpy.save('data/who.npy', self.who)
        numpy.save('data/wih.npy', self.wih)

        return finalOutputs

    def query(self, inputsList):

        #Convert inputs into a 2D Array
        inputs = numpy.array(inputsList,ndmin=2).T
        #Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.wih, inputs)
        #Calculate the output of the hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        #calculate signals into the output layer
        finalInputs = numpy.dot(self.who,hiddenOutputs)
        #Calculate the results
        finalOutputs = self.activationFunction(finalInputs)
        return finalOutputs

