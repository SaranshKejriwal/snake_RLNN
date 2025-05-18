# importing libraries
import pygame
import time
import random
import numpy as np #imported for reliable array comparisons while changing directions
import snakeMaths

class snakeObject:


    snakeStartingDirection = np.array([1,0])

    snakeBody = [[200, 200], [190, 200], [180, 200],  [170, 200], [160,200]] #snake body is of length 4 blocks at the start.
    snakeHeadLocation = np.array([200, 200]) #starting position of the snake on the game board
    #numpy will be used for accurate array comparisons

    #Decision vectors from the neural network - PLACEHOLDERS for now.
    left = np.array([1,0,0]) #turning Left, relative to the current direction of movement
    noAction = np.array([0,1,0]) #no action is equivalent to moving in the same direction as before
    right = np.array([0,0,1]) #turning Left, relative to the current direction of movement

    def __init__(self,gameWindowX, gameWindowY):
        self.resetSnake(gameWindowX,gameWindowY)

    #directionX = 1 #meaning that the snake is moving right, towards +X axis. Left is -1 and 0 is no movement on X
    #directionY = 0 #1 is moving up, -1 is moving down, 0 means no movement on Y axis
    
    #absolute direction vector to indicate the current direction of movement of the snake.
    snakeDirection = np.array([1,0])#corresponds to moving towards the right
    #the decisions of turning left/right will be relative to these 2 values


    #move the snake body without increasing length
    def move(self, isGrowing):

        #move the head as per the current direction of the snake
        self.snakeHeadLocation = np.add(self.snakeHeadLocation, (10*self.getSnakeDirection()))

        #add the next snakePos to the body queue of the snake
        self.snakeBody.insert(0, self.snakeHeadLocation)

        #if the snake hasn't grown, pop the body queue to indicate that the snake has only shifted
        if(not isGrowing):
            self.snakeBody.pop()
            #increase the length of the snake queue by 1 unit ONLY if food is eaten, else increase and reduce together

        return 0


    #ingest the directionVector from the game and turn the snake accordingly
    def changeDirection(self,directionDecisionVector):

        self.snakeDirection = snakeMaths.processDirectionDecision(self.snakeDirection, directionDecisionVector)

        #create a clone of the current decision vector, to avoid reference of a variable to its own latest value
        #currentXDirection = self.snakeDirection[0]#X
        #currentYDirection = self.snakeDirection[1]#Y
        '''
        if (directionDecisionVector == self.left).all():
            self.snakeDirection = snakeMaths.turnSnakeLeft(currentXDirection,currentYDirection)
        elif(directionDecisionVector == self.right).all():
            self.snakeDirection = snakeMaths.turnSnakeRight(currentXDirection,currentYDirection)
        elif(directionDecisionVector != self.noAction).all():
            print("snake could not parse decision vector:", directionDecisionVector)
        '''
        return 0 


    def getSnakeDirection(self):
        #return np.array([self.directionX,self.directionY]) #only one of these should be 1/-1, the other should be 0
        return self.snakeDirection

    def getSnakeHead(self):
        return self.snakeHeadLocation

    def getSnakeBody(self):
        return self.snakeBody

    #in training mode, bring the snake back to its original starting state if the snake dies.
    def resetSnake(self, gameWindowX, gameWindowY):
        self.snakeBody.clear() #clear existing list
        head = [((gameWindowX//20) * 10),((gameWindowY//20) * 10)] #get the center of the game as the head
        self.snakeBody = [[head[0], head[1]], [head[0]-10, head[1]], [head[0]-20, head[1]],  [head[0]-30, head[1]],[head[0]-40,head[1]]] #list reference cannot be done in the same way as np array, hence writing the whole value
        self.snakeHeadLocation = np.array(head)
        self.snakeDirection = self.snakeStartingDirection
        return 0
