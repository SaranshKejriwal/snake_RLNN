'''
In simpleNN, we were trying to force the model to predict the decision with max reward, without giving the model ANY insight on the rewards
This class will attempt to build Q-learning from scratch in python, by calculating Q values
'''
#from ast import Num
from colorsys import yiq_to_rgb
from math import isnan, log
import numpy as np
import snakeMaths

class reinfNN:

    #this will inherit its hyper parameters from the model class. This is only a single neural network
    
    input_layer_neurons = 9 #the state vector has FIXED 9 binary values from the decisionRewardCalculator.getStateVectorForNetwork function

    #Note - 12 neurons has been selected somewhat randomly for now, just to kick things off....can even have hundreds of neurons
    hidden_layer_neurons = 50 #the W1 vector will be a hidden_layer_neurons X input_layer_neurons matrix

    output_layer_neurons = 3 # since output will have 3 dimensional values only - corresponding to turn left, turn right, do nothing
    
    #this is not a hyperParam. THis value will track the loss within this specific network
    currentLoss = 9999 #initialized as a super high value to track any neurons that remain untrained in case their cell was largely populated.

    learning_rate = 0.01 #for updating params
    
    epsilon = 0.999 #Randomness parameter to have high exploration earlier and high exploitation later. This value will reduce as training iterations increase.
    gamma = 0.9 #discount rate for future rewards. 0 indicates that we are interested in quick rewards

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


    def trainModel(self, x_train, y_train):
        
        #numSamples = np.shape(y_train)[0] #y.size will give number of cells, not number of rows.

        Z1, A1, Z2, A2 = self.forwardProp(x_train)
        #Note that in supervised learning, we would provide the expected output value that the model should have achieved. In RL, the loss is defined based on the max reward that the decision could have achieved.
        expectedOutputProbability = self.updateLossValue(A2,y_train)

        #print("loss at cell position ", cellPosition)
        #print(self.currentLoss)

        #reduce learning rate if loss is getting lower
        #self.adaptLearningRate(cellPosition)

        if(isnan(self.currentLoss)): # just for seeing any issues in the logs
            print("Non-numeric Loss found at cell level. Printing parameters:")
            print("Z1 ",Z1)
            print("A1 ",A1)
            print("Z2 ",Z2)
            print("A2 ",A2)

        dW2, dB2, dW1, dB1 = self.backProp(expectedOutputProbability, A1,Z1,self.W2,x_train)

        self.updateParams(dW2, dB2, dW1, dB1)
        '''
        print("Z1:",Z1.shape)
        print("A1:",A1.shape)
        print("Z2:",Z2.shape)
        print("A2:",A2.shape)
        '''
        #print(A2)#without training, this output has a distribution of [0.33, 0.33, 0.33], which is correct

        return self.getDecisionFromPrediction(A2)


    def runModel(self, x_train):
        
        #simply run forward prop without training the model
        Z1, A1, Z2, A2 = self.forwardProp(x_train)
        return self.getDecisionFromPrediction(A2)

    def forwardProp(self, x_train):
        
        Z1 = np.add(self.W1.dot(x_train) , self.b1) #equivalent to Z1 = W1.X1_T + b1; should be of shape (12,m)


        #Normalization required at this step for smoothing. Because Z1 values in just 10 iterations will start to touch infinity
        Z1_max = np.absolute(Z1).max(0,keepdims=True) #for one training example, Z1 is of shape (12,1) -> take the max ABSOLUTE value across all +ive and -ive weights

        Z1_normalized = np.divide(Z1,Z1_max) #this ensures that pre-activations are <1...ensuring that tanh() and softmax() combination does not cause +infinity values (especially in softmax)

        #print("Z1", Z1)
        #print("Z1_max", Z1_max)
        #print("Z1_norm", Z1_normalized)

        A1 = snakeMaths.tanh(Z1_normalized)  #12,m
        
        Z2 = np.add(self.W2.dot(A1) , self.b2) #equivalent to Z2 = W2.A1 + b1     Note that A1 is NOT transposed.

        A2 = snakeMaths.softmax(Z2, softmax_axis=0) #should be of shape (3,m) 

        return Z1_normalized, A1, Z2, A2

    #Entry point - this is where the modelContainer passes the state data to the Model
    def predictNextStep(self,stateVector, reward, isTraining, currentIterationCounter, idealDecision):

        #update epsilon value to select a random outcome with a decreasing probability
        self.epsilon = 1/(1 + (currentIterationCounter/100)) #1/(1+ x/100) will tend towards 0 steadily as x increases

        if isTraining:
            # model is training, so weights will be updated via forward prop and backprop
            # when the model makes a decision, ignore that decision and get the model to explore with a certain epsilon probability.
            # as the iteration counter increases, epsilon tends to 0 
            return self.getRandomExploreDecisionByEpsilon( self.trainModel(stateVector,idealDecision)) #the output of tranModel is already made binary

        else:
            # model is in test. Weights will not be updated and only forward prop will be run
            # Note that the model does not need the Exploration at test phase, so the epsilon parameter will NOT be relevant 
            return self.runModel(stateVector)

        return 0

    def updateLossValue(self, A2, y_train):

        #A2 is of the format [0.4,0.3,0.3] and y_train is already one-Hot -> [0,1,0]

        #get the indices from the predictions, corresponding to the outputs y of interest.
        probabilityOfExpectedOutput = np.multiply(A2,y_train) #get the probability of ONLY the expected output via element-wise multiplication

        #using negative log likelihood method to calculate loss value for all the training examples
        try:
            lossVector = -1 * np.log10(A2)   
            #self.currentLoss = (np.multiply(lossVector,yOneHot.T).sum())/numExamples #get the loss against the indices of the expected output value.
            self.currentLoss = np.sum((np.multiply(lossVector,y_train.T)))

            #print(self.currentLoss)
            #self.currentLoss = (-1 * np.sum(np.log(probabilityOfExpectedOutput)))/numExamples
            #Note - Do NOT attempt an element wise multiplication and THEN take a log of that, because most elements there will be 0, and log(0) is -infinity

        except:
            print("Negative Log Likelihood failed for: ", A2) #needed in case there are any "Not-a-number" issues.      

        return probabilityOfExpectedOutput #this will be used in the BackProp

    def backProp(self, A2y, A1, Z1, W2,x_train):


        dW2 = (-1)*(A2y.dot(A1.T))#check implementation notes -> A2y is (9,m) and A1 is (12,m); dW2 should be (9,12), same as W2
        dA1 = (-1)*((W2.T).dot(A2y)) #this is an intermediate step used to calculate dW1 and dB1;

        dB2 = (-1)*(np.sum(A2y, axis=1, keepdims=True))

        dZ1 = np.multiply(dA1,snakeMaths.dTanh(Z1)) #note that this is an element-wise multiplication, not a dot-product
        #size of dZ1 should be (12,m), same as Z1

        dB1 = np.sum(dA1.dot(snakeMaths.dTanh(Z1.T)), axis = 1, keepdims=True) #equivalent to d(loss)/d(A1) . d(A1)/d(Z1) 
        #d(A1)/d(Z1) = g'(Z1) = dTanh(Z1)
        #note that dB1 = dZ1.sum because Z1 = W1.X + B1; Z1 has distinct columns for each training example but b1 doesn't

        #IMPORTANT - keepdims = True ensures that dB1 is of shape (12,1) and not (12,)

        #print(self.W1.shape)
        dW1 = dZ1.dot(x_train.T)

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

        W1 = np.random.randn(self.hidden_layer_neurons,self.input_layer_neurons) * 0.1 #12 neurons of 81 dimensions, to align with input matrix
        
        #NOTE - unlike Supervised learning, we want the model to be confidently wrong and then learn. Hence we can initialize higher values
        b1 = np.random.randn(self.hidden_layer_neurons,1) #Note - adding (x,y) creates a list of lists.
        #b1 = np.zeros((self.hidden_layer_neurons,1)) #reduced to 0 to remove initialization bias and ensure uniformly random predictions

        W2 = np.random.randn(self.output_layer_neurons,self.hidden_layer_neurons) * 0.1
        
        b2 = np.random.randn(self.output_layer_neurons,1)
        #b2 = np.zeros((self.output_layer_neurons,1)) #reduced to 0 to remove initialization bias and ensure uniformly random predictions

        return W1, b1, W2, b2

    def getRandomExploreDecisionByEpsilon(self, modelDecision):

        #get a random float between 0 and 1. If that is less than epsilon, return a random decision, else return the model decision. 
        #Epsilon will decrease over time, so the probability of getting a random value lower than epsilon will also reduce
        if np.random.rand() < self.epsilon:

            return snakeMaths.getRandomDirectionDecision()
        else:
            return modelDecision

    # a crude way to convert the probability vector into a binary array based on index of max probability
    def getDecisionFromPrediction(self, A2):
        if np.argmax(A2) == 0: #turn left
            return snakeMaths.left
        elif np.argmax(A2) == 1: #no action
            return snakeMaths.noAction
        elif np.argmax(A2) == 2: #turn right
            return snakeMaths.right