import numpy as np
import random

left = np.array([1,0,0]) #turning Left, relative to the current direction of movement
noAction = np.array([0,1,0]) #no action is equivalent to moving in the same direction as before
right = np.array([0,0,1]) #turning Left, relative to the current direction of movement

allDecisions = np.array([left , noAction, right]) #corresponding to Turn Left, Do Nothing and Turn Right

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


#until the neural network is made, this function will be used to randomly get a decision between turn left, turn right or do nothing
def getRandomDirectionDecision():
    #Return any one of the 3 decisions randomly
    return allDecisions[random.randint(0,2)] #return any one element at random

'''
#Copied from old mathFunctions.py_______________
'''

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
    yOneHot = np.zeros((y.size, 9)) #get m x 9 zeroes which will be arranged into 9 columns, hence an (m,3) vector of zeros

    yOneHot[np.arange(y.size), y-1] = 1 #if y = 5, this means that 4th index from 0-4 should be set to 1
    return yOneHot