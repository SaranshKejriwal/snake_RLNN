
import random
import numpy as np

class food:

    #limits of the window size in which the food can spawn randomly.
    xLimit = 720
    yLimit = 480 #based on the size of the gameWindow

    foodLocation = [0,0] #where the food will be on the board once it spawns

    def __init__(self,xLimit,yLimit):
        self.xLimit = xLimit
        self.yLimit = yLimit
        self.foodLocation = np.array([random.randrange(1, (self.xLimit//10)) * 10, random.randrange(1, (self.yLimit//10)) * 10])

    def respawn(self, snakeBody):

        #move the food to another random position
        self.foodLocation = [random.randrange(1, (self.xLimit//10)) * 10, random.randrange(1, (self.yLimit//10)) * 10]

        #ensure once that the food is not spawned on top of the snake itself
        for pos in snakeBody:
                if self.foodLocation[0] == pos[0] and  self.foodLocation[1] == pos[1]:
                    self.foodLocation = [random.randrange(1, (self.xLimit//10)) * 10, random.randrange(1, (self.yLimit//10)) * 10]

    def getFoodLocation(self):
        return self.foodLocation