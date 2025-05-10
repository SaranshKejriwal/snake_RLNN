import random
from xxlimited import foo
import numpy as np

from food import food
from snakeGame import snakeGame
from simpleNN import simpleNN
import decisionRewardCalculator


#this class will create multiple instances of snake games, and a single instance of the neural network

class modelContainer:

    modelNetwork = simpleNN()

    #while the model may play hundreds of games, those will all be sequential, 
    #so only one instance of the game needs to be created and then reinitialized
    game = snakeGame()

    #this iteration counter will be useful to reduce the epsilon parameter in the network. As iterations increase, we want more exploitation, less exploration
    currentIterationCounter = 0

    def trainModel(self):

        #self.game.startGame(self, isTraining = True)
        print("Starting model training...")
              
        #instead of running an external for-loop for iterations, if the snake dies in training mode, it simply respawns at the starting point and body
        self.game.startGame(self, isTraining = True)


    def testModel(self):
        self.game.startGame(self, isTraining = False) #model already trained - params will not update weights


    #this will be called post training - somewhat redundant at the moment
    def startGame(self):
        self.game.startGame(self, isTraining=False) 
        #pass the model as an input to the game, for the game to call the next decision funtion

    def getCurrentModel(self):
        return self.modelNetwork

    def predictNextStep(self, gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation, currentReward, isTraining):

        
        #this is a deterministic calculation of reward in all possible decisions and selecting the best outcome, based PURELY on the immediate next step.
        #return decisionRewardCalculator.getHighestRewardDecision(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation)
        #note that this rewardCalculator will not be capable of any emergent properties - it will only look at the payoff of the next direction, so the snake can trap itself.

        #this is the neural network that we want to train.
        return self.modelNetwork.predictNextStep(decisionRewardCalculator.getStateVectorForNetwork(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation), currentReward, isTraining)


        #this is a test code to run the initial model
        #return self.getRandomDirectionDecision()

    #until the neural network is made, this function will be used to randomly get a decision between turn left, turn right or do nothing
    def getRandomDirectionDecision(self):

        allDecisions = np.array([[1,0,0] , [0,1,0], [0,0,1]]) #corresponding to Turn Left, Do Nothing and Turn Right
        #note that [0,0,0] is not a valid case because the neural network should be predicting one of these 3 decisions with a certain higher probability than others.

        return allDecisions[random.randint(0,2)] #return any one element at random

