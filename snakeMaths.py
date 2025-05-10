import numpy as np


#evaluate the current Direction, and turn the snake LEFT, relative to that direction.
def turnSnakeLeft(currentXDirection, currentYDirection):
    '''
    Truth Table of (x,y):
    [0,1] > [-1,0]; 	Up to Left
    [0,-1] > [1,0]; 	Down to Right
    [-1,0] > [0,-1]; 	Left to Down
    [1,0] > [0,1]; 	    Right to Up

    '''

    return np.array([-1 * currentYDirection, currentXDirection])

    #evaluate the current Direction, and turn the snake RIGHT, relative to that direction.
def turnSnakeRight(currentXDirection, currentYDirection):
    '''
    Truth Table of (x,y):
    [0,1] > [1,0]; 	Up to Right
    [0,-1] > [-1,0]; 	Down to Left
    [-1,0] > [0,1]; 	Left to Up
    [1,0] > [0,-1]; 	Right to Down

    '''

    return np.array([currentYDirection, -1 * currentXDirection])


#check if the game is lost because the snake bit itself or the map boundary walls
def isGameOver(snakeHead, snakeBody, gameWindowX, gameWindowY):

    #touching the walls
    if snakeHead[0] < 0 or snakeHead[0] > gameWindowX-10:
        return True
    if snakeHead[1] < 0 or snakeHead[1] > gameWindowY-10:
        return True

    # Touching the snake body
    for block in snakeBody[1:]: #index starts at 1, not 0; Head itself is excluded
        if (block == snakeHead).all():
            return True
    return False

#this method will convert the input parameters into a normalized state vector for the model to ingest
def getStateVectorForNetwork(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection, foodLocation, currentReward, isTraining):
        
    dangerVector = decisionRewardCalculator.getDangerVector(gameWindowX, gameWindowY, snakeHead, snakeBody, snakeDirection)

    stateVector = np.array([
            
        snakeHead[0]/gameWindowX, #snakeHead X coordinate - normalized to (0,1) by dividing with gameWindowX
        snakeHead[0]/gameWindowX,

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
    - currentReward can be divided by 1000 for normalization - will always be negative
    '''

    return stateVector

#Copied from old mathFunctions.py

def tanh(z):
    return np.tanh(z)

def dTanh(tanh):
    #Derivative of tanh() is 1- tanh()^2
    return 1 - np.square(tanh)

def softmax(z, softmax_axis = 0):
    #z_sum = np.sum(np.exp(z),axis = softmax_axis, keepdims=True)
    #print("z_sum Shape:", z_sum.shape)
    A2 = np.divide(np.exp(z),(np.sum(np.exp(z),axis = softmax_axis, keepdims=True))) 
    #axis=0 argument is important, to ensure that the sum is only along the preactivations of ONE training example, not against the ENTIRE dataset. 
    return A2

#this function returns a list, or list of lists, of random values between -0.5 and 0.5
#xDim is the size of the outer list and yDim is the siz of the inner list.
def getGaussianInit(xDim, yDim):
    return np.random.randn(xDim,yDim)

#this takes a digit from 1-9 as an input and returns an array of zeroes such that only the yth value is 1
def getOneHotVector(y):
    yOneHot = np.zeros((y.size, 9)) #get m x 9 zeroes which will be arranged into 9 columns, hence an (m,9) vector of zeros

    yOneHot[np.arange(y.size), y-1] = 1 #if y = 5, this means that 4th index from 0-4 should be set to 1
    return yOneHot