'''
This class will contain the Qtable for each of the combinations of states that the game can be in, and actions that the snake can take.
'''
from curses.ascii import TAB
import numpy as np
import snakeMaths

class QTableContainer:

    epsilon = 1 #start with randomly exploring all possibilities and decay the randomness
    epsilonDecay = 0.9995 #epsilon will reduce by a geometric progression of 0.995, not an arithmetic progression like before.
    lowestEpsilon = 0.01 #lower threshold for randomness

    learning_rate = 0.9 #used to update the new Q values.     
    gamma = 0.95 #discount factor
    
    actionCount = 3 #this will always be 3 because the snake only has 3 possible decision outcomes.

    def __init__(self, windowX, windowY):

        cellsX = windowX // 10 #get number of cells after dividing by cell size
        cellsY = windowY // 10 #get number of cells after dividing by cell size

        #this is where all rewards for all actions in all states will be stored.
        self.qTable = np.zeros([cellsX, cellsY, 4, 8, cellsX,cellsY,3])
        #all values will be initialized at 0 and then adjusted over time.

        #print('Q table shape:',self.qTable.shape)

        '''
        In a 100x100 size window, there are 10x10 cells for the head and food.

        The state is defined by: 
        -headLocation (10 x 10 possible values)
        -direction (4 possible values)
        -Danger vector (8 possible values)
        -foodLocation (10 x 10 possible values) 

        Action will always have 3 possible values.

        Therefore, the qTable will be of size 10 x 10 x 4 x 8 x 10 x 10 x 3 = 960,000 values

        '''
    
        return

    def predictNextStep(self, snakeHead, snakeDirection, dangerVector, foodLocation ):

        directionIndex = self.getQTableDirectionIndex(snakeDirection)
        dangerIndex = self.getQTableDangerIndex(dangerVector)

        #return the action with the max 
        modelDecisionIndex = np.argmax(self.qTable[snakeHead[0]//10, snakeHead[1]//10,directionIndex, dangerIndex,foodLocation[0]//10, foodLocation[1]//10])
        modelDecision = self.getDirectionFromIndex(modelDecisionIndex)

        return self.getRandomExploreDecisionByEpsilon(modelDecision)


    def getRandomExploreDecisionByEpsilon(self, modelDecision):

        #decay the epsilon value of the active instance of the Q table container.
        if(self.epsilon > self.lowestEpsilon):
            self.epsilon = self.epsilon * self.epsilonDecay #reduce the probability of getting a random output

        #get a random float between 0 and 1. If that is less than epsilon, return a random decision, else return the model decision. 
        #Epsilon will decrease over time, so the probability of getting a random value lower than epsilon will also reduce
        if np.random.rand() < self.epsilon:

            return snakeMaths.getRandomDirectionDecision()
        else:
            return modelDecision

    #this is supposed to be the important function which will update the reward values in the table, 
    #the reward value update will be as per the random explorations that the snake is able to do, AND for different locations of the food
    def updateQTable(self):
        pass



    #very primitive function to convert a 2D direction vector into an array index
    def getQTableDirectionIndex(self, snakeDirection):

        #order of indices is Right, Left, Up, Down
        if (snakeDirection == snakeMaths.snakeMovingRight).all():
            return 0
        elif (snakeDirection == snakeMaths.snakeMovingLeft).all():
            return 1
        elif (snakeDirection == snakeMaths.snakeMovingUp).all():
            return 2
        elif (snakeDirection == snakeMaths.snakeMovingDown).all():
            return 3

    #dangerVector is expected to be a 3D binary array at all times
    def getQTableDangerIndex(self, dangerVector):
        tableIndex = dangerVector[0] + 2*dangerVector[1] + 4*dangerVector[2]
        return tableIndex


    def getDirectionFromIndex(self, maxRewardIndex):

        if maxRewardIndex == 0: #reward for turning left is highest
            return snakeMaths.left 
        elif maxRewardIndex == 1: #reward for not turning is highest
            return snakeMaths.noAction
        elif maxRewardIndex == 2:#reward for turning right is highest
            return snakeMaths.right

        return 0 #unreachable code ideally