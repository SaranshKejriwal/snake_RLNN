import numpy as np
import snakeMaths

#this class will determine the choice that returns the max reward, to train the neural network. It will also serve as a deterministic approach to the snake game.


#Decision vectors from the neural network - PLACEHOLDERS for now.
left = np.array([1,0,0]) #turning Left, relative to the current direction of movement
noAction = np.array([0,1,0]) #no action is equivalent to moving in the same direction as before
right = np.array([0,0,1]) #turning Left, relative to the current direction of movement


def getHighestRewardDecision(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation):

    rewardVector = getRewardVector(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation)
    maxRewardIndex = np.argmax(rewardVector)

    if maxRewardIndex == 0: #reward for turning left is highest
        return np.array([1,0,0]) #self.left
    elif maxRewardIndex == 1: #reward for not turning is highest
        return np.array([0,1,0]) #self.noAction
    elif maxRewardIndex == 2:#reward for turning right is highest
        return np.array([0,0,1])# self.right 

    return 0 #unreachable code ideally

#this will return a 1x3 vector of the payoff of all 3 decisions -> [left, no_action, right]. 
#this will only be called internally, and  will return the numeric reward based on distance from food, and death
def getRewardVector(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation):
    rewardVector = np.array([0,0,0])#each index corresponding to Left/No_Action/Right; Initialized at 0

    #reward for turning left
    rewardVector[0] = getRewardFromDirection(gameWindowX, gameWindowY, snakeBody, foodLocation, snakeNextHead = np.add(snakeHead, (10 * snakeMaths.turnSnakeLeft(snakeDirection[0],snakeDirection[1])))) #turnLeft was called

    #reward for no turning
    rewardVector[1] = getRewardFromDirection(gameWindowX, gameWindowY, snakeBody, foodLocation, snakeNextHead = np.add(snakeHead, (10 * (snakeDirection)))) #snake continued in current direction

    #reward for turning right
    rewardVector[2] = getRewardFromDirection(gameWindowX, gameWindowY, snakeBody, foodLocation, snakeNextHead = np.add(snakeHead, (10 * snakeMaths.turnSnakeRight(snakeDirection[0],snakeDirection[1])))) #turnRight was called

    return rewardVector #this function will be called directly when loss needs to be calculated

#this will return a 1x3 vector of which decisions will lead to Death of the snake
#this will only be called internally, and will return a binary vector - Index is True on the decision where snake dies, and False where snake lives
def getDangerVector(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection):
    dangerVector = np.array([0,0,0])#each index corresponding to Left/No_Action/Right; Initialized at 0

    #reward for turning left
    dangerVector[0] = snakeMaths.isGameOver( np.add(snakeHead, (10 * snakeMaths.turnSnakeLeft(snakeDirection[0],snakeDirection[1]))), snakeBody, gameWindowX, gameWindowY) #turnLeft was called. Would snake die?

    #reward for no turning
    dangerVector[1] = snakeMaths.isGameOver(np.add(snakeHead, (10 * (snakeDirection))), snakeBody, gameWindowX, gameWindowY) #snake continued in current direction. Would snake die?

    #reward for turning right
    dangerVector[2] = snakeMaths.isGameOver(np.add(snakeHead, (10 * snakeMaths.turnSnakeRight(snakeDirection[0],snakeDirection[1]))), snakeBody, gameWindowX, gameWindowY) #turnRight was called. Would snake die?

    return dangerVector #this function will be called directly when loss needs to be calculated

def getRewardFromDirection(gameWindowX, gameWindowY, snakeBody, foodLocation, snakeNextHead):
        
    reward = 0 #initialized at 0, but will always be negative. The largest reward should be selected

    distanceFromFood = np.linalg.norm(snakeNextHead - foodLocation) #this calculates the euclidean distance between the next head position and the food

    reward -= distanceFromFood #the higher the distance, the lower the net reward

    if snakeMaths.isGameOver(snakeNextHead,snakeBody, gameWindowX, gameWindowY):
        reward -= 1000 #reduce 1000 from the reward if the snake dies. Assumption is that 1000 is comparable to the max Euclidean distance of 680, in a 480x480 board

    return reward

#this method will convert the input parameters into a normalized state vector for the model to ingest
def getStateVectorForNetwork(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation):
        
    dangerVector = getDangerVector(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection)

    stateVector = np.array([
            
        snakeHead[0]/gameWindowX, #snakeHead X coordinate - normalized to (0,1) by dividing with gameWindowX
        snakeHead[1]/gameWindowY,

        snakeDirection[0], #X coordinate of direction
        snakeDirection[1], #Y coordinate of direction

        dangerVector[0], 
        dangerVector[1],
        dangerVector[2], #dangerVector needs to be expanded like this to ensure that the overall input to the network is 1-dimensional and does not become an array of arrays

        foodLocation[0]/gameWindowX,#foodLocation X coordinate - normalized to (0,1) by dividing with gameWindowX
        foodLocation[1]/gameWindowY
            
        ])
        
    '''
    Define the state.
    State can have the following values
    - snakeHead X coordinate - normalized to (0,1) by dividing with gameWindowX
    - snakeHead Y coordinate - normalized to (0,1) by dividing with gameWindowY

    - NOT REQUIRED - snakeBody coordinates - This is covered in the dangerVector 
        
    - DirectionX - direct ingestion of 1/-1 value
    - DirectionY - direct ingestion of 1/-1 value

    - Danger directions - is it dangerous to turn left from current position? or turn right? or go straight? - THIS will account for the body of the snake

    - foodLocation X coordinate - normalized to (0,1) by dividing with gameWindowX
    - foodLocation Y coordinate - normalized to (0,1) by dividing with gameWindowY
    '''
    return stateVector