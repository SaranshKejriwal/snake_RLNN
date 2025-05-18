from pyexpat import model
import random
import numpy as np

from food import food
from snakeGame import snakeGame
from trainingDataContainer import trainingDataContainer
from simpleNN import simpleNN
from reinfNN import reinfNN
from QTableContainer import QTableContainer
import decisionRewardCalculator
import snakeMaths

#this class will create multiple instances of snake games, and a single instance of the neural network

class modelContainer:

    simpleNeuralNetwork = simpleNN()

    #while the model may play hundreds of games, those will all be sequential, 
    #so only one instance of the game needs to be created and then reinitialized
    game = snakeGame()

    #this iteration counter will be useful to reduce the epsilon parameter in the network. As iterations increase, we want more exploitation, less exploration
    currentIterationCounter = 0

    #this defines the number of datapoints that we want in the training set
    dataGatheringIterationLimit = 10000

    #this defines the number of iterations for which the model should be trained
    trainingIterationLimit = 10000

    #this data container will actually store the training data from the deterministic model to train our simple Neural network.
    #Note that this should ideally not be needed for reinforcement learning, if the idea is for the model to LEARN how to play the game.
    dataContainer = trainingDataContainer()

    def __init__(self,trainingDataContainer, lastGatheringIteration):
        self.dataContainer = trainingDataContainer
        self.dataGatheringIterationLimit = lastGatheringIteration
        self.qTableContainer = QTableContainer(self.game.getWindowX(), self.game.getWindowY())

    def startGatheringData(self):
        print('Training Data collection started...')
        self.game.startGame(self, True, self.dataGatheringIterationLimit) #isTraining is set to True

    def trainModel(self, trainingIterations):

        #self.game.startGame(self, isTraining = True)
        print("Starting model training...")

        x_train = np.array(self.dataContainer.getStateVectorsData())
        y_train = np.array(self.dataContainer.getDecisionsData())

        #print(x_train.shape)
        #print(y_train.shape)
        
        #instead of running an external for-loop for iterations, if the snake dies in training mode, it simply respawns at the starting point and body
        #self.game.startGame(self, isTraining = True)
        self.simpleNeuralNetwork.trainModel(x_train, y_train ,trainingIterations)


    def testModel(self):
        self.game.startGame(self, False, self.dataGatheringIterationLimit) #model already trained


    #this will be called post training - somewhat redundant at the moment
    def startGame(self):
        self.game.startGame(self, isTraining=False) 
        #pass the model as an input to the game, for the game to call the next decision funtion

    #this will be called once the training data has been gathered
    def quitGame(self):
        self.game.quitGame()

    def getCurrentModel(self):
        return self.simpleNeuralNetwork

    def predictNextStep(self, gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation, currentReward, isTraining):
        
        #use the newly created Q table to get the action with the best reward.
        return self.qTableContainer.predictNextStep(snakeHead, snakeDirection, decisionRewardCalculator.getDangerVector(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection), foodLocation)
        
        #ignoring the neural network models
        '''
        #increase the IterationCounter to stop the data gather, OR to reduce the epsilon with each iteration in the reinforcement learning model
        self.currentIterationCounter += 1

        #print iteration log
        if self.currentIterationCounter % 1000 == 0:
            print('Model Container Iterations completed: ', self.currentIterationCounter)

        #this is a deterministic calculation of reward in all possible decisions and selecting the best outcome, based PURELY on the immediate next step.
        #note that this rewardCalculator will not be capable of any emergent properties - it will only look at the payoff of the next direction, so the snake can trap itself.

        #increase the iteration counter by 1, to reduce the probability of making random decisions as the iterations increase - applies to reinfNN only

        #store data in container and return ideal decision, and then train the model to start returning the model predictions.
        if self.currentIterationCounter < self.dataGatheringIterationLimit:
            #get the ideal highest decision to feed to the model - this seems like standard supervised learning.
            idealDecision = decisionRewardCalculator.getHighestRewardDecision(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation)

            self.dataContainer.addDatapoint(decisionRewardCalculator.getNormalizedStateVectorForNetwork(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation), idealDecision)
            return idealDecision

        else:
            #this is the simple neural network that we want to train.
            modelDecision =  self.simpleNeuralNetwork.predictNextStep(decisionRewardCalculator.getNormalizedStateVectorForNetwork(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation))
            print('model Decision:', modelDecision)
            return modelDecision

        #this is a test code to run the initial model
        #return snakeMaths.getRandomDirectionDecision()
        '''


