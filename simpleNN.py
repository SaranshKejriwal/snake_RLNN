'''
This class contains a single 10 layer neural network.
The simpleNNModel will create 81 of these simpleNN objects, one for each cell in Sudoku.
'''
#from ast import Num
from colorsys import yiq_to_rgb
from math import isnan, log
import numpy as np
import snakeMaths

class simpleNN:

    #this will inherit its hyper parameters from the model class. This is only a single neural network
    
    input_layer_neurons = 9

    #Note - 10 neurons has been selected somewhat randomly for now, just to kick things off....
    hidden_layer_neurons = 10 #the W1 vector will be a hidden_layer_neurons X input_layer_neurons matrix

    output_layer_neurons = 3 # since output will have 3 dimensional values only - corresponding to turn left, turn right, do nothing
    
    #this is not a hyperParam. THis value will track the loss within this specific network
    currentLoss = 9999 #initialized as a super high value to track any neurons that remain untrained in case their cell was largely populated.

    learning_rate = 0.01 #for updating params
    
    epsilon = 0.001 #Randomness parameter to have high exploration earlier and high exploitation later.
    gamma = 0 #discount rate for future rewards. 0 indicates that we are interested in quick rewards

    def __init__(self):
        #constructor

        #initialize the weights and biases of this network with random values from -0.5 to 0.5
        self.W1, self.b1, self.W2, self.b2 = self.initParams()

        '''
        W1 is 12x81
        b1 is 12x1
        W2 is 9x12
        b2 is 9x1
        '''

        return


    def trainModel(self, x_train, y_train, cellPosition):
        
        numSamples = np.shape(y_train)[0] #y.size will give number of cells, not number of rows.

        Z1, A1, Z2, A2 = self.forwardProp(x_train)
        expectedOutputProbability = self.updateLossValue(A2,y_train, numSamples)

        #print("loss at cell position ", cellPosition)
        #print(self.currentLoss)

        #reduce learning rate if loss is getting lower
        self.adaptLearningRate(cellPosition)

        if(isnan(self.currentLoss)): # just for seeing any issues in the logs
            print("Non-numeric Loss found at cell level. Printing parameters:")
            print("Cell position: ", cellPosition)
            print("Z1 ",Z1)
            print("A1 ",A1)
            print("Z2 ",Z2)
            print("A2 ",A2)

        dW2, dB2, dW1, dB1 = self.backProp(numSamples, expectedOutputProbability, A1,Z1,self.W2,x_train)

        self.updateParams(dW2, dB2, dW1, dB1)

        return 

    def forwardProp(self, x_train):
        
        Z1 = np.add(self.W1.dot(x_train.T) , self.b1) #equivalent to Z1 = W1.X1_T + b1; should be of shape (12,m)

        #Normalization required at this step for smoothing. Because Z1 values in just 10 iterations will start to touch infinity
        Z1_max = np.absolute(Z1).max(0,keepdims=True) #for one training example, Z1 is of shape (12,1) -> take the max ABSOLUTE value across all +ive and -ive weights

        Z1_normalized = np.divide(Z1,Z1_max) #this ensures that pre-activations are <1...ensuring that tanh() and softmax() combination does not cause +infinity values (especially in softmax)

        #print("Z1", Z1)
        #print("Z1_max", Z1_max)
        #print("Z1_norm", Z1_normalized)

        A1 = snakeMaths.tanh(Z1_normalized)  #12,m
        
        Z2 = np.add(self.W2.dot(A1) , self.b2) #equivalent to Z2 = W2.A1 + b1     Note that A1 is NOT transposed.

        #should be of shape (9,m)
        A2 = snakeMaths.softmax(Z2, softmax_axis=0) #should be of shape (9,m) 

        return Z1_normalized, A1, Z2, A2

    def predictNextStep(self,, isTraining):




        if isTraining:
            # model is training, so weights will be updated via forward prop and backprop
            return

        else:
            # model is in test. Weights will not be updated and only forward prop will be run
            return

        return 0

    def updateLossValue(self, A2, y_train,numSamples):

        #Note - Each neural network applies to an individual cell of the sudoku, so Loss will actually be an 81-dimension array

        yOneHot = snakeMaths.getOneHotVector(y_train)

        #get the indices from the predictions, corresponding to the outputs y of interest.
        probabilityOfExpectedOutput = np.multiply(A2,yOneHot.T) #get the probability of ONLY the expected output via element-wise multiplication

        #using negative log likelihood method to calculate loss value for all the training examples
        try:
            lossVector = -1 * np.log10(A2)   
            #self.currentLoss = (np.multiply(lossVector,yOneHot.T).sum())/numExamples #get the loss against the indices of the expected output value.
            self.currentLoss = np.sum((np.multiply(lossVector,yOneHot.T)))/numSamples

            #self.currentLoss = (-1 * np.sum(np.log(probabilityOfExpectedOutput)))/numExamples
            #Note - Do NOT attempt an element wise multiplication and THEN take a log of that, because most elements there will be 0, and log(0) is -infinity

        except:
            print("Negative Log Likelihood failed for: ", A2) #needed in case there are any "Not-a-number" issues.      

        return probabilityOfExpectedOutput

    def backProp(self, numSamples, A2y, A1, Z1, W2,x_train):

        #Refer to implementation notes in Word document. probabilityOfExpectedOutput is equivalent to A2y

        dW2 = (-1)*(A2y.dot(A1.T))/numSamples #check implementation notes -> A2y is (9,m) and A1 is (12,m); dW2 should be (9,12), same as W2
        dA1 = (-1)*((W2.T).dot(A2y))/numSamples #this is an intermediate step used to calculate dW1 and dB1; dA1 should be (12,2), same as A1

        dB2 = (-1)*(np.sum(A2y, axis=1, keepdims=True))/numSamples # dB2 should be (9,1), same as B2

        dZ1 = np.multiply(dA1,snakeMaths.dTanh(Z1)) #note that this is an element-wise multiplication, not a dot-product
        #size of dZ1 should be (12,m), same as Z1

        dB1 = np.sum(dA1.dot(snakeMaths.dTanh(Z1.T)), axis = 1, keepdims=True) #equivalent to d(loss)/d(A1) . d(A1)/d(Z1) 
        #d(A1)/d(Z1) = g'(Z1) = dTanh(Z1)
        #note that dB1 = dZ1.sum because Z1 = W1.X + B1; Z1 has distinct columns for each training example but b1 doesn't

        #IMPORTANT - keepdims = True ensures that dB1 is of shape (12,1) and not (12,)

        dW1 = dZ1.dot(x_train)

        return dW2, dB2, dW1, dB1

    #this method updates params after backprop
    def updateParams(self,dW2, dB2, dW1, dB1):

        self.W2 = self.W2 - self.learning_rate * dW2 #should be of shape (9,12)
        self.b2 = self.b2 - self.learning_rate * dB2 #should be of shape (9,1)
        self.W1 = self.W1 - self.learning_rate * dW1 #should be of shape (12,81)
        self.b1 = self.b1 - self.learning_rate * dB1 #should be of shape (12,1)

        return

    #this method reduces the learning rate for each cell's network after its loss is low enough to not require large jumps
    def adaptLearningRate(self, cellPosition):

        if(self.currentLoss < 0.3 and self.learning_rate == 0.01):
            self.learning_rate = 0.001 #reduced to one tenth for slower progression
            #print("Learning Rate reduction for cell position ",cellPosition)
            #print("Reducing learning rate to ", self.learning_rate)

        return
    #note - we should eventually add code to compare current loss with previous loss also.

    #this method initializes the starting weights and biases of the network before training.
    def initParams(self):

        #Note - we want to curb the initialization loss by ensuring that the initial random weights don't end up taking extreme values that are "confidently wrong". Initial biases can be set to 0.
        #Weights will multiplied by 0.1 to ensure that the initialization of the weights is as close to 0 as possible, and the network is basically making random guesses without training.

        W1 = np.random.randn(self.hidden_layer_neurons,self.input_layer_neurons) * 0.01 #12 neurons of 81 dimensions, to align with input matrix
        
        #b1 = np.random.randn(self.input_layer_neurons,1) #Note - adding (x,y) creates a list of lists.
        b1 = np.zeros((self.hidden_layer_neurons,1))

        W2 = np.random.randn(self.output_layer_neurons,self.hidden_layer_neurons) * 0.01
        
        #b2 = np.random.randn(self.hidden_layer_neurons,1)
        b2 = np.zeros((self.output_layer_neurons,1))

        #Note - Loss will start increasing if Learning rate is too high

        return W1, b1, W2, b2

