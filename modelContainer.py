import random
import numpy as np

from food import food
from snakeGame import snakeGame
from simpleNN import simpleNN
import decisionRewardCalculator
import snakeMaths

#this class will create multiple instances of snake games, and a single instance of the neural network

class modelContainer:

    modelNetwork = simpleNN()

    #while the model may play hundreds of games, those will all be sequential, 
    #so only one instance of the game needs to be created and then reinitialized
    game = snakeGame()

    #this iteration counter will be useful to reduce the epsilon parameter in the network. As iterations increase, we want more exploitation, less exploration
    currentIterationCounter = 0

    #storing all decisions this way ensures that no new objects are created at runtime


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

        #increase the iteration counter by 1, to reduce the probability of making random decisions as 


        #this is the simple neural network that we want to train.

        #get the ideal highest decision to feed to the model - this seems like standard supervised learning.
        idealDecision = decisionRewardCalculator.getHighestRewardDecision(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation).reshape(3,1)
        
        #increase the IterationCounter to reduce the epsilon with each iteration
        self.currentIterationCounter += 1

                #print iteration log
        if self.currentIterationCounter % 10000 == 0:
            print('Iterations: ', self.currentIterationCounter)

        return self.modelNetwork.predictNextStep(decisionRewardCalculator.getStateVectorForNetwork(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation), currentReward, isTraining, self.currentIterationCounter, idealDecision)
        



        #this is a test code to run the initial model
        #return snakeMaths.getRandomDirectionDecision()



