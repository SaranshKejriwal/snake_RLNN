'''
This class will contain the Qtable for each of the combinations of states that the game can be in, and actions that the snake can take.
'''
import numpy as np
import snakeMaths

class QTableContainer:

    epsilon = 1 #start with randomly exploring all possibilities and decay the randomness
    epsilonDecay = 0.99995 #epsilon will reduce by a geometric progression of 0.995, not an arithmetic progression like before.
    #Note that a decay factor of 0.9995 drops to <1% in just 10k iterations, 
    #This means that the snake will end up spinning in circles because that's the only path discovered.
    
    lowestEpsilon = 0.01 #lower threshold for randomness

    alpha = 0.9 #used to update the new Q values.     
    gamma = 0.75 #discount factor
    
    actionCount = 3 #this will always be 3 because the snake only has 3 possible decision outcomes.

    #these will track the latest decision taken by the model in any state. 
    #It will be referred while updating table
    latestModelDecisionIndex = 0


    def __init__(self, windowX, windowY):

        cellsX = windowX // 10 #get number of cells after dividing by cell size
        cellsY = windowY // 10 #get number of cells after dividing by cell size

        #this is where all rewards for all actions in all states will be stored.
        #self.qTable = np.ones([cellsX, cellsY, 4, 8, cellsX,cellsY,3]) * (-500)
        self.qTable = np.zeros([cellsX, cellsY, 4, 8, cellsX,cellsY,3])
        #all values will be initialized at -500 and then adjusted over time to less negative values.
        #Note - We cannot initialzie the QTable at zeros, because all reward values are negative.
        #Since there is no positive reward, zero will always seem like the most rewarding option

        print('Q table shape:',self.qTable.shape)

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

    def predictNextStep(self, stateVector):

        #Directly pass the vector as an index to the table and return the index with the max reward
        self.latestModelDecisionIndex = np.argmax(self.qTable[tuple(stateVector)])
        modelDecision = self.getDirectionFromIndex(self.latestModelDecisionIndex)

        #latestModelDecisionIndex will be referred later when the Q table is being updated.
        #Note that because of epsilon, the model will be more likely to explore and update multiple Q-values for the future.

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

    #this is supposed to be the IMPORTANT function which will update the reward values in the table, 
    #the reward value update will be as per the random explorations that the snake is able to do, AND for different locations of the food
    def updateQTable(self, currentStateVector, nextStateVector, modelDecision, rewardFromDecision):
        #modelContainer instance will be calling this function, once the decision is taken.


        #took an action given a state. We use this to update the QValue, if initialized at 0, using Bellman equation.
        oldQValue = self.qTable[tuple(currentStateVector)][self.latestModelDecisionIndex]

        #print('current state vector:',currentStateVector)
        #print(currentStateVector.shape)
        #print(nextStateVector.shape)
        #print(self.qTable.shape)
        #print('current state reward vector:',self.qTable[tuple(currentStateVector)])

        #print('next state vector:',nextStateVector)
        #Need to update the Q value of the Next state, with the max Q value in the next state.

        #print('next state reward vector:',self.qTable[tuple(nextStateVector)])
        nextStateMaxReward = np.max(self.qTable[tuple(nextStateVector), :]) 
        
        #print ('old Q value',oldQValue)
        #print('next state max Reward',nextStateMaxReward)

        #print('reward from decision', rewardFromDecision)
        

        #update the Q table using the Bellman equation:
        self.qTable[tuple(currentStateVector)][self.latestModelDecisionIndex] = (1 - self.alpha) * oldQValue + self.alpha *(rewardFromDecision + self.gamma * nextStateMaxReward)
        
        #print('updated reward vector:',self.qTable[tuple(currentStateVector)])
        #print('\n')

    def getDirectionFromIndex(self, maxRewardIndex):

        if maxRewardIndex == 0: #reward for turning left is highest
            return snakeMaths.left 
        elif maxRewardIndex == 1: #reward for not turning is highest
            return snakeMaths.noAction
        elif maxRewardIndex == 2:#reward for turning right is highest
            return snakeMaths.right

        return 0 #unreachable code ideally